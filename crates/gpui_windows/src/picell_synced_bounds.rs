//! [PiCell] `PlatformWindow::set_bounds_with_frame`: the window moves together with the first
//! frame drawn at its new size.
//!
//! `set_bounds` calls `SetWindowPos`, and the frame at the new size comes after a full redraw. The
//! window is DirectComposition (`WS_EX_NOREDIRECTIONBITMAP`), so until that frame is presented DWM
//! shows the last one at the old size pinned to the new rectangle's top-left corner and leaves the
//! rest transparent. A window sized by its content — PiCell's photo preview going from a portrait
//! to a landscape photo — showed what is behind it for as long as the redraw took (35–50 ms in a
//! debug build, every DWM frame captured with Desktop Duplication), and a shrinking one showed the
//! old frame cut off.
//!
//! Here the order is turned around. GPUI asks from its frame callback (`fast::synced_bounds`):
//! the swap chain takes the new size at once while the window stays where it is (DWM keeps
//! showing the last presented frame, `ResizeBuffers` does not clear it), GPUI draws its next frame
//! at that size, and the window moves right after that frame is presented, in the same callback.
//! DWM takes the frame and the new rectangle together only when both land before the same
//! composition, and in a release build they did not always: a frame the GPU had not finished yet
//! came one composition after the new rectangle (1 move in 7), and a composition falling between
//! a finished frame and the move showed the new frame in the old rectangle (3 in 24). So that
//! frame is presented once the GPU has finished it and a composition has just passed, which
//! leaves the move a whole refresh period (0 in 24).
//!
//! Nothing here applies to a window that is minimized, hidden, maximized or fullscreen, without
//! DirectComposition, or whose size is not changing (a pure move has no gap): those take
//! `set_bounds`. A plain `set_bounds`, or a size change from anywhere else (the user, a
//! maximize), drops a move still waiting.

use gpui::{Bounds, DevicePixels, Pixels, Size};
use gpui_util::ResultExt as _;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HWND, RECT},
        Graphics::{Dwm::DwmFlush, Dxgi::IDXGIDevice2},
        System::Threading::{CreateEventW, WaitForSingleObject},
        UI::WindowsAndMessaging::{
            GetClientRect, IsIconic, IsWindowVisible, IsZoomed, SWP_NOACTIVATE, SWP_NOZORDER,
            SetWindowPos,
        },
    },
    core::{Interface as _, PCWSTR},
};

use crate::{WindowsWindowInner, directx_renderer::DirectXRenderer};

/// How long the moving frame waits for the GPU at most, in milliseconds.
const GPU_WAIT_LIMIT_MS: u32 = 50;

/// A move waiting for its frame.
#[derive(Clone, Copy)]
pub(crate) struct PendingBounds {
    /// The window rectangle to set, border included.
    rect: RECT,
    /// The client size the swap chain and GPUI were given.
    size: Size<DevicePixels>,
}

/// Whether the renderer's next present is the frame the window moves with.
#[derive(Default)]
pub(crate) struct SyncedPresent(bool);

/// Gives the swap chain the new size and keeps the move for after the next present: `false`
/// when this window takes a plain `set_bounds` instead.
pub(crate) fn request(
    window: &WindowsWindowInner,
    hwnd: HWND,
    bounds: Bounds<Pixels>,
    rect: RECT,
) -> bool {
    let scale = window.state.scale_factor.get();
    let size = bounds.to_device_pixels(scale).size;
    let unusual = unsafe {
        !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() || IsZoomed(hwnd).as_bool()
    } || window.state.is_fullscreen();
    let composited = window
        .state
        .renderer
        .try_borrow()
        .is_ok_and(|renderer| renderer.direct_composition.is_some());
    // GPUI's size too: one left behind by a dropped move is put back here.
    let same_size = client_size(hwnd) == Some(size)
        && window.state.logical_size.get() == size.to_pixels(scale);
    if unusual || !composited || same_size {
        return false;
    }
    window.state.logical_size.set(size.to_pixels(scale));
    crate::fast::device_recovery::resize(window, size);
    window.state.renderer.borrow_mut().picell_synced_present = SyncedPresent(true);
    window
        .state
        .pending_bounds
        .set(Some(PendingBounds { rect, size }));
    true
}

/// Forwarded to by `DirectXRenderer::present`: the frame the window moves with is presented once
/// the GPU has finished it and a composition has just passed.
pub(crate) fn before_present(renderer: &mut DirectXRenderer) {
    if !std::mem::take(&mut renderer.picell_synced_present.0) {
        return;
    }
    wait_for_gpu(renderer);
    unsafe { DwmFlush() }.log_err();
}

/// Forwarded to by `draw_window` after the frame callback, which took the request and presented
/// the frame at the new size.
pub(crate) fn apply_after_present(window: &WindowsWindowInner, hwnd: HWND) {
    let Some(pending) = window.state.pending_bounds.get() else {
        return;
    };
    cancel(window);
    let unusual = unsafe { IsIconic(hwnd).as_bool() || IsZoomed(hwnd).as_bool() }
        || window.state.is_fullscreen();
    if unusual {
        return;
    }
    let RECT {
        left,
        top,
        right,
        bottom,
    } = pending.rect;
    unsafe {
        SetWindowPos(
            hwnd,
            None,
            left,
            top,
            right - left,
            bottom - top,
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
    }
    .log_err();
}

/// `WM_SIZE` with a size other than the pending one: someone else resized the window.
pub(crate) fn on_size(window: &WindowsWindowInner, size: Size<DevicePixels>) {
    if window
        .state
        .pending_bounds
        .get()
        .is_some_and(|pending| pending.size != size)
    {
        cancel(window);
    }
}

/// Drops a waiting move, and the wait its frame would have had.
pub(crate) fn cancel(window: &WindowsWindowInner) {
    window.state.pending_bounds.set(None);
    if let Ok(mut renderer) = window.state.renderer.try_borrow_mut() {
        renderer.picell_synced_present = SyncedPresent(false);
    }
}

/// Waits until the GPU has finished the work submitted so far: DWM composes a frame only once
/// it is finished.
fn wait_for_gpu(renderer: &DirectXRenderer) {
    let Some(device) = renderer
        .devices
        .as_ref()
        .and_then(|devices| devices.device.cast::<IDXGIDevice2>().log_err())
    else {
        return;
    };
    let Some(event) = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }.log_err() else {
        return;
    };
    if unsafe { device.EnqueueSetEvent(event) }.log_err().is_some() {
        unsafe { WaitForSingleObject(event, GPU_WAIT_LIMIT_MS) };
    }
    unsafe { CloseHandle(event) }.log_err();
}

fn client_size(hwnd: HWND) -> Option<Size<DevicePixels>> {
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.ok()?;
    Some(Size {
        width: DevicePixels(rect.right - rect.left),
        height: DevicePixels(rect.bottom - rect.top),
    })
}

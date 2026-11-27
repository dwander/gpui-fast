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
//! Here the order is turned around. The swap chain and GPUI take the new size while the window
//! stays where it is: DWM keeps showing the last presented frame (`ResizeBuffers` does not clear
//! it). The window moves right after the first frame at that size is presented, so DWM takes the
//! frame and the new rectangle together.
//!
//! Nothing here applies to a window that is minimized, hidden, maximized or fullscreen, without
//! DirectComposition, or whose client size is not changing (a pure move has no gap): those take
//! `set_bounds`. A size change that arrives meanwhile from anywhere else (the user, a maximize)
//! drops the pending move.

use std::rc::Rc;

use gpui::{Bounds, DevicePixels, Pixels, Size};
use gpui_util::ResultExt as _;
use windows::Win32::{
    Foundation::{HWND, RECT},
    UI::WindowsAndMessaging::{
        GetClientRect, IsIconic, IsWindowVisible, IsZoomed, SWP_NOACTIVATE, SWP_NOZORDER,
        SetWindowPos,
    },
};

use crate::{WindowsWindowInner, directx_renderer::DirectXRenderer};

/// A move waiting for its frame.
#[derive(Clone, Copy)]
pub(crate) struct PendingBounds {
    /// The window rectangle to set, border included.
    rect: RECT,
    /// The client size the swap chain and GPUI were given.
    size: Size<DevicePixels>,
    /// The swap chain's present count right after it took the new size.
    present_count: Option<u32>,
}

/// Starts a synced move: `false` when this window takes a plain `set_bounds` instead.
pub(crate) fn request(
    window: &Rc<WindowsWindowInner>,
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
    if unusual || !composited || client_size(hwnd) == Some(size) {
        return false;
    }
    // Not from inside the caller: it is usually drawing (a window sized by its content decides
    // its size while rendering), and GPUI cannot take a resize while the app is borrowed.
    let inner = window.clone();
    window
        .executor
        .spawn(async move {
            inner.handle_size_change(size, scale, true);
            let present_count = present_count(&inner.state.renderer.borrow());
            inner.state.pending_bounds.set(Some(PendingBounds {
                rect,
                size,
                present_count,
            }));
        })
        .detach();
    true
}

/// After a frame was drawn: moves the window if a frame at the pending size has been presented.
pub(crate) fn apply_after_present(window: &WindowsWindowInner, hwnd: HWND) {
    let Some(pending) = window.state.pending_bounds.get() else {
        return;
    };
    let presented = window
        .state
        .renderer
        .try_borrow()
        .ok()
        .and_then(|renderer| present_count(&renderer));
    if presented.is_some() && presented == pending.present_count {
        return;
    }
    window.state.pending_bounds.set(None);
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
        window.state.pending_bounds.set(None);
    }
}

fn client_size(hwnd: HWND) -> Option<Size<DevicePixels>> {
    let mut rect = RECT::default();
    unsafe { GetClientRect(hwnd, &mut rect) }.ok()?;
    Some(Size {
        width: DevicePixels(rect.right - rect.left),
        height: DevicePixels(rect.bottom - rect.top),
    })
}

fn present_count(renderer: &DirectXRenderer) -> Option<u32> {
    let resources = renderer.resources.as_ref()?;
    unsafe { resources.swap_chain.GetLastPresentCount() }.ok()
}

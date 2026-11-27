//! [PiCell] `Window::set_bounds_with_frame`: a window sized by its content changes size together
//! with the first frame drawn at the new size.
//!
//! A request waits here until GPUI's frame callback takes it ([`draw`]). Taken before drawing,
//! the frame is drawn at the new size. Asked for while drawing — a window that decides its size in
//! `render` — the frame just drawn was laid out at the old size, so it is drawn again at the new
//! one instead of being presented; presented, the new content showed at the old size for a few
//! frames before the window followed (a release build, every move). The platform gives the
//! drawable the new size at once and moves the window after presenting
//! (`PlatformWindow::set_bounds_with_frame`, gpui_windows' `picell_synced_bounds`).

use crate::{App, ArenaClearNeeded, Bounds, Pixels, Window};

/// The request the next frame takes.
#[derive(Default)]
pub(crate) struct SyncedBounds(Option<Bounds<Pixels>>);

impl Window {
    /// [`Window::set_bounds`] that moves the window together with the first frame drawn at the
    /// new size instead of before it — for a window sized by its content, where the gap between
    /// the two shows as a flash. Takes effect at the next frame, and may be asked for from
    /// `render`. See [`crate::PlatformWindow::set_bounds_with_frame`].
    pub fn set_bounds_with_frame(&mut self, bounds: Bounds<Pixels>) {
        self.synced_bounds.0 = Some(bounds);
        // A frame, not `refresh`: asked every frame for the bounds the window already has, a
        // refresh would rebuild every view every frame.
        self.invalidator.set_dirty(true);
    }
}

/// Forwarded to by `Window::set_bounds` and `Window::resize`: a later plain move wins.
pub(crate) fn cancel(window: &mut Window) {
    window.synced_bounds.0 = None;
}

/// Forwarded to by the frame callback in place of `Window::draw`.
pub(crate) fn draw(window: &mut Window, cx: &mut App) -> ArenaClearNeeded {
    take(window, cx);
    let arena_clear_needed = window.draw(cx);
    if !take(window, cx) {
        return arena_clear_needed;
    }
    arena_clear_needed.clear(cx);
    window.draw(cx)
}

/// Forwarded to by the end of `Window::draw`: only the frame callback takes a request, and a draw
/// outside it (a key dispatch) has just marked the window clean.
pub(crate) fn keep_frame_coming(window: &Window) {
    if window.synced_bounds.0.is_some() {
        window.invalidator.set_dirty(true);
    }
}

/// Hands a waiting request to the platform: `true` when the drawable already has the new size,
/// so GPUI takes it too and the next frame is drawn at it.
fn take(window: &mut Window, cx: &mut App) -> bool {
    let Some(bounds) = window.synced_bounds.0.take() else {
        return false;
    };
    let resized = window.platform_window.set_bounds_with_frame(bounds);
    if resized {
        window.bounds_changed(cx);
    }
    resized
}

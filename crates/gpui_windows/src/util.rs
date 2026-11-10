use std::{collections::HashMap, sync::OnceLock};

use anyhow::Context;
use gpui_util::ResultExt;
use windows::{
    UI::{
        Color,
        ViewManagement::{UIColorType, UISettings},
    },
    Win32::{
        Foundation::*,
        Graphics::Dwm::*,
        Graphics::Gdi::{
            BITMAPINFO, BITMAPINFOHEADER, CreateBitmap, CreateDIBSection, DIB_RGB_COLORS,
            DeleteObject, HGDIOBJ, MONITOR_DEFAULTTONEAREST, MonitorFromPoint,
        },
        System::LibraryLoader::LoadLibraryA,
        UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI},
        UI::WindowsAndMessaging::*,
    },
    core::{BOOL, PCSTR},
};

use crate::*;
use gpui::*;

pub(crate) trait HiLoWord {
    fn hiword(&self) -> u16;
    fn loword(&self) -> u16;
    fn signed_hiword(&self) -> i16;
    fn signed_loword(&self) -> i16;
}

impl HiLoWord for WPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

impl HiLoWord for LPARAM {
    fn hiword(&self) -> u16 {
        ((self.0 >> 16) & 0xFFFF) as u16
    }

    fn loword(&self) -> u16 {
        (self.0 & 0xFFFF) as u16
    }

    fn signed_hiword(&self) -> i16 {
        ((self.0 >> 16) & 0xFFFF) as i16
    }

    fn signed_loword(&self) -> i16 {
        (self.0 & 0xFFFF) as i16
    }
}

pub(crate) unsafe fn get_window_long(hwnd: HWND, nindex: WINDOW_LONG_PTR_INDEX) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        GetWindowLongPtrW(hwnd, nindex)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        GetWindowLongW(hwnd, nindex) as isize
    }
}

pub(crate) unsafe fn set_window_long(
    hwnd: HWND,
    nindex: WINDOW_LONG_PTR_INDEX,
    dwnewlong: isize,
) -> isize {
    #[cfg(target_pointer_width = "64")]
    unsafe {
        SetWindowLongPtrW(hwnd, nindex, dwnewlong)
    }
    #[cfg(target_pointer_width = "32")]
    unsafe {
        SetWindowLongW(hwnd, nindex, dwnewlong as i32) as isize
    }
}

pub(crate) fn windows_credentials_target_name(url: &str) -> String {
    format!("zed:url={}", url)
}

pub(crate) fn load_cursor(style: CursorStyle) -> Option<HCURSOR> {
    static ARROW: OnceLock<SafeCursor> = OnceLock::new();
    static IBEAM: OnceLock<SafeCursor> = OnceLock::new();
    static CROSS: OnceLock<SafeCursor> = OnceLock::new();
    static HAND: OnceLock<SafeCursor> = OnceLock::new();
    static SIZEWE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENS: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENWSE: OnceLock<SafeCursor> = OnceLock::new();
    static SIZENESW: OnceLock<SafeCursor> = OnceLock::new();
    static NO: OnceLock<SafeCursor> = OnceLock::new();
    let (lock, name) = match style {
        CursorStyle::IBeam | CursorStyle::IBeamCursorForVerticalLayout => (&IBEAM, IDC_IBEAM),
        CursorStyle::Crosshair => (&CROSS, IDC_CROSS),
        CursorStyle::PointingHand | CursorStyle::DragLink => (&HAND, IDC_HAND),
        CursorStyle::ResizeLeft
        | CursorStyle::ResizeRight
        | CursorStyle::ResizeLeftRight
        | CursorStyle::ResizeColumn => (&SIZEWE, IDC_SIZEWE),
        CursorStyle::ResizeUp
        | CursorStyle::ResizeDown
        | CursorStyle::ResizeUpDown
        | CursorStyle::ResizeRow => (&SIZENS, IDC_SIZENS),
        CursorStyle::ResizeUpLeftDownRight => (&SIZENWSE, IDC_SIZENWSE),
        CursorStyle::ResizeUpRightDownLeft => (&SIZENESW, IDC_SIZENESW),
        CursorStyle::OperationNotAllowed => (&NO, IDC_NO),
        _ => (&ARROW, IDC_ARROW),
    };
    Some(
        *(*lock.get_or_init(|| {
            HCURSOR(
                unsafe { LoadImageW(None, name, IMAGE_CURSOR, 0, 0, LR_DEFAULTSIZE | LR_SHARED) }
                    .log_err()
                    .unwrap_or_default()
                    .0,
            )
            .into()
        })),
    )
}

/// A registered custom image cursor: the source image plus one `HCURSOR` per monitor DPI.
///
/// 커서 비트맵은 화면 화소 그대로 그려진다 — 32px 원본을 그대로 쓰면 150%·200% 화면에서 시스템
/// 커서(배율만큼 커진다)보다 작아진다. 그래서 포인터가 있는 모니터의 배율로 키운 커서를 배율마다
/// 한 번 만들어 둔다. 핸들은 앱 수명 동안 유지한다 (종료 시 암묵 해제).
pub(crate) struct CustomCursor {
    pub(crate) image: CustomCursorImage,
    pub(crate) by_dpi: HashMap<u32, HCURSOR>,
}

impl CustomCursor {
    /// `dpi` 화면용 커서 — 처음이면 만든다. 만들 수 없는 이미지면 `None`.
    pub(crate) fn at_dpi(&mut self, dpi: u32) -> Option<HCURSOR> {
        if let Some(&cursor) = self.by_dpi.get(&dpi) {
            return Some(cursor);
        }
        let scale = dpi as f32 / USER_DEFAULT_SCREEN_DPI as f32;
        let cursor = if (scale - 1.0).abs() < 0.01 {
            create_custom_cursor(&self.image)?
        } else {
            create_custom_cursor(&scale_cursor_image(&self.image, scale))?
        };
        self.by_dpi.insert(dpi, cursor);
        Some(cursor)
    }
}

/// 마우스 포인터가 있는 모니터의 유효 DPI — 못 읽으면 기본값(배율 1.0).
pub(crate) fn dpi_at_pointer() -> u32 {
    let mut point = POINT::default();
    if unsafe { GetCursorPos(&mut point) }.is_err() {
        return USER_DEFAULT_SCREEN_DPI;
    }
    let monitor = unsafe { MonitorFromPoint(point, MONITOR_DEFAULTTONEAREST) };
    let (mut dpi_x, mut dpi_y) = (0, 0);
    match unsafe { GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y) } {
        Ok(()) if dpi_x > 0 => dpi_x,
        _ => USER_DEFAULT_SCREEN_DPI,
    }
}

/// 커서 그림을 `scale` 배로 키운다 — 원본 화소를 네모 칸 그대로 키우고, 칸 경계가 대상 화소
/// 중간에 걸리는 곳만 면적 비율로 섞는다. 정수 배율이면 최근접과 같아 도트가 그대로 살고, 1.5배
/// 같은 배율에서도 최근접처럼 선 굵기가 들쭉날쭉해지지 않는다. 핫스팟은 같은 원본 화소를 가리킨다.
pub(crate) fn scale_cursor_image(image: &CustomCursorImage, scale: f32) -> CustomCursorImage {
    let (src_w, src_h) = (image.width as usize, image.height as usize);
    let dst_w = ((src_w as f32 * scale).round() as usize).max(1);
    let dst_h = ((src_h as f32 * scale).round() as usize).max(1);
    let cols = pixel_coverage(src_w, dst_w);
    let rows = pixel_coverage(src_h, dst_h);
    let mut rgba = vec![0u8; dst_w * dst_h * 4];
    for (dy, row) in rows.iter().enumerate() {
        for (dx, col) in cols.iter().enumerate() {
            // 알파를 곱한 채로 섞어야 투명한 화소의 색이 가장자리에 번지지 않는다.
            let mut acc = [0.0f32; 4];
            for &(sy, wy) in row {
                for &(sx, wx) in col {
                    let i = (sy * src_w + sx) * 4;
                    let alpha = image.rgba[i + 3] as f32 * wx * wy;
                    for c in 0..3 {
                        acc[c] += image.rgba[i + c] as f32 * alpha;
                    }
                    acc[3] += alpha;
                }
            }
            if acc[3] > 0.0 {
                let o = (dy * dst_w + dx) * 4;
                for c in 0..3 {
                    rgba[o + c] = (acc[c] / acc[3]).round().min(255.0) as u8;
                }
                rgba[o + 3] = acc[3].round().min(255.0) as u8;
            }
        }
    }
    let hotspot = |hot: u32, src: usize, dst: usize| {
        (((hot as f32 + 0.5) * dst as f32 / src as f32) as u32).min(dst as u32 - 1)
    };
    CustomCursorImage {
        rgba,
        width: dst_w as u32,
        height: dst_h as u32,
        hot_x: hotspot(image.hot_x, src_w, dst_w),
        hot_y: hotspot(image.hot_y, src_h, dst_h),
    }
}

/// 대상 화소마다 겹치는 원본 화소와 그 면적 비율 (한 대상 화소의 비율 합은 1).
fn pixel_coverage(src: usize, dst: usize) -> Vec<Vec<(usize, f32)>> {
    let span = src as f32 / dst as f32;
    (0..dst)
        .map(|d| {
            let (lo, hi) = (d as f32 * span, (d + 1) as f32 * span);
            (lo.floor() as usize..src)
                .take_while(|&s| (s as f32) < hi)
                .filter_map(|s| {
                    let overlap = hi.min(s as f32 + 1.0) - lo.max(s as f32);
                    (overlap > 1e-4).then_some((s, overlap / span))
                })
                .collect()
        })
        .collect()
}

/// Builds an `HCURSOR` from a [`CustomCursorImage`] (RGBA8, top-down) with a hotspot.
/// Returns `None` on invalid input or any GDI failure. The returned cursor is owned by the
/// caller (kept for the app lifetime; freed implicitly at process exit).
pub(crate) fn create_custom_cursor(image: &CustomCursorImage) -> Option<HCURSOR> {
    let w = image.width as i32;
    let h = image.height as i32;
    if w <= 0 || h <= 0 || image.rgba.len() < (image.width as usize * image.height as usize * 4) {
        return None;
    }
    unsafe {
        // 32bpp top-down DIB (컬러) — biHeight 음수 = top-down.
        let bmi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: 0, // BI_RGB
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let color = CreateDIBSection(None, &bmi, DIB_RGB_COLORS, &mut bits, None, 0).log_err()?;
        if bits.is_null() {
            let _ = DeleteObject(HGDIOBJ(color.0));
            return None;
        }
        // RGBA → BGRA 로 DIB 에 복사 (Windows 32bpp 비트맵은 BGRA).
        let dst = std::slice::from_raw_parts_mut(bits as *mut u8, (w * h * 4) as usize);
        for (i, px) in image.rgba.chunks_exact(4).take((w * h) as usize).enumerate() {
            let o = i * 4;
            dst[o] = px[2]; // B
            dst[o + 1] = px[1]; // G
            dst[o + 2] = px[0]; // R
            dst[o + 3] = px[3]; // A
        }
        // 모노크롬 AND 마스크 (전부 0 = 알파 채널로 투명 처리).
        let mask_stride = (((w + 15) / 16) * 2) as usize;
        let mask_bits = vec![0u8; mask_stride * h as usize];
        let mask = CreateBitmap(w, h, 1, 1, Some(mask_bits.as_ptr() as *const _));
        if mask.0.is_null() {
            let _ = DeleteObject(HGDIOBJ(color.0));
            return None;
        }
        let icon_info = ICONINFO {
            fIcon: BOOL(0), // FALSE → 커서(핫스팟 사용)
            xHotspot: image.hot_x,
            yHotspot: image.hot_y,
            hbmMask: mask,
            hbmColor: color,
        };
        let hicon = CreateIconIndirect(&icon_info);
        // CreateIconIndirect 가 비트맵을 복사하므로 원본은 즉시 해제 가능.
        let _ = DeleteObject(HGDIOBJ(color.0));
        let _ = DeleteObject(HGDIOBJ(mask.0));
        Some(HCURSOR(hicon.log_err()?.0))
    }
}

/// This function is used to configure the dark mode for the window built-in title bar.
pub(crate) fn configure_dwm_dark_mode(hwnd: HWND, appearance: WindowAppearance) {
    let dark_mode_enabled: BOOL = match appearance {
        WindowAppearance::Dark | WindowAppearance::VibrantDark => true.into(),
        WindowAppearance::Light | WindowAppearance::VibrantLight => false.into(),
    };
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark_mode_enabled as *const _ as _,
            std::mem::size_of::<BOOL>() as u32,
        )
        .log_err();
    }
}

#[inline]
pub(crate) fn logical_point(x: f32, y: f32, scale_factor: f32) -> Point<Pixels> {
    Point {
        x: px(x / scale_factor),
        y: px(y / scale_factor),
    }
}

// https://learn.microsoft.com/en-us/windows/apps/desktop/modernize/apply-windows-themes
#[inline]
pub(crate) fn system_appearance() -> Result<WindowAppearance> {
    let ui_settings = UISettings::new()?;
    let foreground_color = ui_settings.GetColorValue(UIColorType::Foreground)?;
    // If the foreground is light, then is_color_light will evaluate to true,
    // meaning Dark mode is enabled.
    if is_color_light(&foreground_color) {
        Ok(WindowAppearance::Dark)
    } else {
        Ok(WindowAppearance::Light)
    }
}

#[inline(always)]
fn is_color_light(color: &Color) -> bool {
    ((5 * color.G as u32) + (2 * color.R as u32) + color.B as u32) > (8 * 128)
}

pub(crate) fn with_dll_library<R, F>(dll_name: PCSTR, f: F) -> Result<R>
where
    F: FnOnce(HMODULE) -> Result<R>,
{
    let library = unsafe {
        LoadLibraryA(dll_name).with_context(|| format!("Loading dll: {}", dll_name.display()))?
    };
    let result = f(library);
    unsafe {
        FreeLibrary(library)
            .with_context(|| format!("Freeing dll: {}", dll_name.display()))
            .log_err();
    }
    result
}

#[cfg(test)]
mod cursor_scale_tests {
    use super::scale_cursor_image;
    use gpui::CustomCursorImage;

    const CLEAR: [u8; 4] = [0, 0, 0, 0];
    const RED: [u8; 4] = [255, 0, 0, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    /// 가로 `pixels` 한 줄짜리 커서.
    fn row(pixels: &[[u8; 4]], hot_x: u32) -> CustomCursorImage {
        CustomCursorImage {
            rgba: pixels.concat(),
            width: pixels.len() as u32,
            height: 1,
            hot_x,
            hot_y: 0,
        }
    }

    fn pixel(image: &CustomCursorImage, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * image.width + x) * 4) as usize;
        image.rgba[i..i + 4].try_into().unwrap()
    }

    /// 정수 배율은 최근접과 같다 — 도트가 섞이지 않고 칸 그대로 커진다.
    #[test]
    fn integer_scale_keeps_the_dots() {
        let out = scale_cursor_image(&row(&[RED, CLEAR], 1), 2.0);
        assert_eq!((out.width, out.height), (4, 2));
        for y in 0..2 {
            assert_eq!(pixel(&out, 0, y), RED);
            assert_eq!(pixel(&out, 1, y), RED);
            assert_eq!(pixel(&out, 2, y), CLEAR);
            assert_eq!(pixel(&out, 3, y), CLEAR);
        }
        // 핫스팟은 같은 원본 화소(1번)를 가리킨다.
        assert_eq!((out.hot_x, out.hot_y), (3, 1));
    }

    /// 1.5배 — 원본 칸 경계가 걸린 대상 화소만 반반 섞이고, 투명 쪽 색은 번지지 않는다.
    #[test]
    fn fractional_scale_blends_only_the_straddling_pixel() {
        let out = scale_cursor_image(&row(&[WHITE, CLEAR], 0), 1.5);
        assert_eq!(out.width, 3);
        assert_eq!(pixel(&out, 0, 0), WHITE);
        let half = pixel(&out, 1, 0);
        assert_eq!(half[..3], WHITE[..3], "투명 화소의 검정이 번졌다");
        assert!((127..=128).contains(&half[3]), "반반이어야 할 알파 {}", half[3]);
        assert_eq!(pixel(&out, 2, 0)[3], 0);
    }

    /// 32px 커서가 150%·200% 화면에서 시스템 커서와 같은 크기(48·64)가 된다.
    #[test]
    fn a_32px_cursor_follows_the_system_cursor_size() {
        let image = CustomCursorImage {
            rgba: vec![255; 32 * 32 * 4],
            width: 32,
            height: 32,
            hot_x: 13,
            hot_y: 13,
        };
        let out = scale_cursor_image(&image, 1.5);
        assert_eq!((out.width, out.height, out.hot_x), (48, 48, 20));
        let out = scale_cursor_image(&image, 2.0);
        assert_eq!((out.width, out.height, out.hot_x), (64, 64, 27));
        assert!(out.rgba.iter().all(|&v| v == 255));
    }
}

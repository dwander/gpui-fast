//! What a display offers for HDR output: whether Windows HDR is on, how bright SDR white is and
//! the panel's peak. [`DirectXRenderer::set_hdr_content`] reads the display under its window to
//! choose between the 8-bit sRGB swap chain and the FP16 scRGB one; apps that tone-map images for
//! the display read [`hdr_displays`], so both see the same numbers.
//!
//! [`DirectXRenderer::set_hdr_content`]: crate::DirectXRenderer::set_hdr_content

use gpui::DisplayId;
use windows::{
    Win32::{
        Devices::Display::{
            DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,
            DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME, DISPLAYCONFIG_DEVICE_INFO_HEADER,
            DISPLAYCONFIG_DEVICE_INFO_TYPE, DISPLAYCONFIG_MODE_INFO, DISPLAYCONFIG_PATH_INFO,
            DISPLAYCONFIG_SDR_WHITE_LEVEL, DISPLAYCONFIG_SOURCE_DEVICE_NAME,
            DisplayConfigGetDeviceInfo, GetDisplayConfigBufferSizes, QDC_ONLY_ACTIVE_PATHS,
            QueryDisplayConfig,
        },
        Foundation::{ERROR_SUCCESS, HWND, LUID},
        Graphics::{
            Dxgi::{
                Common::DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, CreateDXGIFactory1,
                DXGI_OUTPUT_DESC1, IDXGIFactory1, IDXGIOutput6,
            },
            Gdi::{MONITOR_DEFAULTTONEAREST, MonitorFromWindow},
        },
    },
    core::Interface,
};

/// Luminance of scRGB `1.0`. DXGI's linear FP16 colour space pins SDR white here; Windows'
/// "SDR content brightness" slider then lifts the white every other window gets.
pub(crate) const SCRGB_WHITE_NITS: f32 = 80.0;

/// `DISPLAYCONFIG_SDR_WHITE_LEVEL::SDRWhiteLevel` counts thousandths of [`SCRGB_WHITE_NITS`].
const SDR_WHITE_LEVEL_PER_SCRGB_WHITE: f32 = 1000.0;

/// A display's HDR capabilities as Windows reports them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrDisplay {
    /// Windows HDR is on for this display — its output colour space is BT.2100 PQ.
    pub hdr_enabled: bool,
    /// SDR white in nits: where UI white has to land in scRGB to match every other window.
    pub sdr_white_nits: f32,
    /// Peak luminance the panel reports, in nits (0 when it reports none).
    pub max_nits: f32,
}

impl HdrDisplay {
    /// The scRGB value of SDR white.
    pub fn sdr_white_scale(&self) -> f32 {
        self.sdr_white_nits / SCRGB_WHITE_NITS
    }

    /// How many times brighter than SDR white the display can go (its peak over SDR white) — the
    /// headroom HDR images are tone-mapped into. `1.0` while Windows HDR is off.
    pub fn headroom(&self) -> f32 {
        if self.hdr_enabled && self.max_nits > self.sdr_white_nits {
            self.max_nits / self.sdr_white_nits
        } else {
            1.0
        }
    }
}

/// Every display DXGI sees, by gpui [`DisplayId`]. Reads afresh each call (a new DXGI factory and
/// a display configuration query), so callers that ask every frame should throttle it.
pub fn hdr_displays() -> Vec<(DisplayId, HdrDisplay)> {
    let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else {
        return Vec::new();
    };
    let white_levels = sdr_white_levels();
    output_descs(&factory)
        .iter()
        .map(|desc| {
            (
                DisplayId::new(desc.Monitor.0 as u64),
                hdr_display(desc, &white_levels),
            )
        })
        .collect()
}

/// Reads the display `hwnd` is on, or `None` when no DXGI output matches its monitor.
///
/// `factory` must be current (`IDXGIFactory1::IsCurrent`): DXGI snapshots each output's colour
/// space when the factory is created, so a stale factory keeps reporting the HDR state from before
/// the user toggled it.
pub(crate) fn probe(factory: &IDXGIFactory1, hwnd: HWND) -> Option<HdrDisplay> {
    let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
    let desc = output_descs(factory)
        .into_iter()
        .find(|desc| desc.Monitor == monitor)?;
    Some(hdr_display(&desc, &sdr_white_levels()))
}

fn hdr_display(desc: &DXGI_OUTPUT_DESC1, white_levels: &[(Vec<u16>, f32)]) -> HdrDisplay {
    let name = trim_nul(&desc.DeviceName);
    HdrDisplay {
        hdr_enabled: desc.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
        // Without a reading, assume the scRGB default rather than guessing a brighter white.
        sdr_white_nits: white_levels
            .iter()
            .find_map(|(gdi_name, nits)| (gdi_name == name).then_some(*nits))
            .unwrap_or(SCRGB_WHITE_NITS),
        max_nits: desc.MaxLuminance,
    }
}

/// Every DXGI output, on whichever adapter drives it.
fn output_descs(factory: &IDXGIFactory1) -> Vec<DXGI_OUTPUT_DESC1> {
    let mut descs = Vec::new();
    // `DXGI_ERROR_NOT_FOUND` past the last adapter / output ends each loop.
    for adapter_index in 0.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(adapter_index) }) else {
            break;
        };
        for output_index in 0.. {
            let Ok(output) = (unsafe { adapter.EnumOutputs(output_index) }) else {
                break;
            };
            if let Ok(output) = output.cast::<IDXGIOutput6>()
                && let Ok(desc) = unsafe { output.GetDesc1() }
            {
                descs.push(desc);
            }
        }
    }
    descs
}

/// SDR white of every active display, by GDI device name (`\\.\DISPLAYn`).
fn sdr_white_levels() -> Vec<(Vec<u16>, f32)> {
    let mut path_count = 0;
    let mut mode_count = 0;
    if unsafe {
        GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &mut path_count, &mut mode_count)
    }
    .is_err()
    {
        return Vec::new();
    }
    let mut paths = vec![DISPLAYCONFIG_PATH_INFO::default(); path_count as usize];
    let mut modes = vec![DISPLAYCONFIG_MODE_INFO::default(); mode_count as usize];
    if unsafe {
        QueryDisplayConfig(
            QDC_ONLY_ACTIVE_PATHS,
            &mut path_count,
            paths.as_mut_ptr(),
            &mut mode_count,
            modes.as_mut_ptr(),
            None,
        )
    }
    .is_err()
    {
        return Vec::new();
    }
    paths.truncate(path_count as usize);

    paths
        .iter()
        .filter_map(|path| {
            let mut source = DISPLAYCONFIG_SOURCE_DEVICE_NAME {
                header: info_header::<DISPLAYCONFIG_SOURCE_DEVICE_NAME>(
                    DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,
                    path.sourceInfo.adapterId,
                    path.sourceInfo.id,
                ),
                ..Default::default()
            };
            if unsafe { DisplayConfigGetDeviceInfo(&mut source.header) } != ERROR_SUCCESS.0 as i32 {
                return None;
            }
            // The white level belongs to the target (the monitor), not the source.
            let mut white = DISPLAYCONFIG_SDR_WHITE_LEVEL {
                header: info_header::<DISPLAYCONFIG_SDR_WHITE_LEVEL>(
                    DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,
                    path.targetInfo.adapterId,
                    path.targetInfo.id,
                ),
                SDRWhiteLevel: 0,
            };
            if unsafe { DisplayConfigGetDeviceInfo(&mut white.header) } != ERROR_SUCCESS.0 as i32
                || white.SDRWhiteLevel == 0
            {
                return None;
            }
            Some((
                trim_nul(&source.viewGdiDeviceName).to_vec(),
                white.SDRWhiteLevel as f32 / SDR_WHITE_LEVEL_PER_SCRGB_WHITE * SCRGB_WHITE_NITS,
            ))
        })
        .collect()
}

/// Header for a `DisplayConfigGetDeviceInfo` request packet of type `T`.
fn info_header<T>(
    kind: DISPLAYCONFIG_DEVICE_INFO_TYPE,
    adapter_id: LUID,
    id: u32,
) -> DISPLAYCONFIG_DEVICE_INFO_HEADER {
    DISPLAYCONFIG_DEVICE_INFO_HEADER {
        r#type: kind,
        size: std::mem::size_of::<T>() as u32,
        adapterId: adapter_id,
        id,
    }
}

fn trim_nul(name: &[u16]) -> &[u16] {
    &name[..name.iter().position(|&c| c == 0).unwrap_or(name.len())]
}

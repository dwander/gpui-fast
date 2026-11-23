//! [PiCell] Photos drawn without the rounded clip on drivers that draw nothing through it.
//!
//! gpui-fast wraps every fragment shader in its rounded clip (`fast/rounded_clip_shader_build.rs`).
//! On an AMD Vega iGPU (Ryzen 5 5600G, reported against PiCell 0.7.5) every photo went missing
//! while the quads, text and badges around them still drew: the driver takes the wrapped
//! `polychrome_sprite_fragment` without complaint and draws nothing with it, and Direct3D has no
//! way to say so. So once per GPU this draws a sprite through it into a small texture, plain and
//! inside nested rounded clips, and reads the pixels back. Where the sprite is missing, polychrome
//! sprites switch to the unwrapped copy (`UNCLIPPED_POLYCHROME_ENTRY`, the shader as it was before
//! gpui-fast): they show, but no rounded clip applies to them. That is photos, and also colour
//! emoji and composited scroll layer tiles, which the same pipeline draws: inside a rounded
//! container they get square corners. Quads, text and paths keep the clip. Nothing is shown to the
//! user; the outcome goes to the log.
//!
//! `GPUI_POLYCHROME_CLIP=0` takes the unwrapped copy without probing (to see the fallback on a GPU
//! that passes), `=1` keeps the wrapped shader without probing.
//!
//! The verdict holds for the adapter and driver version it was measured on: a renderer created
//! later applies it without drawing again, and one recreated after a lost device probes again only
//! if the device came back on another adapter or driver. A probe that fails to run decides nothing,
//! so the next renderer probes again.

use std::borrow::Cow;
use std::slice;
use std::time::Instant;

use anyhow::{Context as _, Result};
use gpui::{
    AtlasKey, Bounds, ContentMask, Corners, DevicePixels, ImageId, PlatformAtlas as _,
    PolychromeSprite, RenderImageParams, ScaledPixels, Scene, TransformationMatrix, point, size,
};
use gpui_util::ResultExt as _;
use parking_lot::Mutex;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_RENDER_TARGET, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_VIEWPORT,
        ID3D11Buffer, ID3D11Device, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
        ID3D11Texture2D,
    },
    Dxgi::{Common::DXGI_SAMPLE_DESC, IDXGIAdapter1, IDXGIDevice},
};
use windows::core::Interface as _;

use crate::DirectXAtlas;
use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::directx_renderer::{
    DirectXGlobalElements, DirectXRenderPipelines, DirectXRenderer, RENDER_TARGET_FORMAT,
    create_fragment_shader, read_back_texture,
};
use crate::fast::frame::{FrameState, Target, draw_scene, draw_scene_masked, upload};

/// Overrides the probe: `0` draws photos without the rounded clip, `1` with it.
const OVERRIDE_ENV: &str = "GPUI_POLYCHROME_CLIP";
/// Side of the square texture the probe draws into, in pixels.
const TARGET_SIZE: u32 = 32;
/// Side of the probe image: small, so the atlas packs it rather than giving it a mipmapped texture
/// of its own, as it does thumbnails.
const IMAGE_SIZE: i32 = 8;
/// Inset of the sprite from the target's edges, so that its centre sits clear of the clips'
/// rounded corners.
const SPRITE_INSET: f32 = 4.0;
/// Corner radius of the probe's clips.
const CLIP_RADIUS: f32 = 6.0;
/// Rounded clips the sprite is drawn inside: nested, as a photo in a rounded cell in a rounded
/// panel is.
const NESTED_CLIPS: usize = 2;
/// The probe image's colour, RGBA. Red and blue match, so the BGRA target reads the same either way.
const COLOR: [u8; 4] = [200, 60, 200, 255];
/// How far a channel may stray from [`COLOR`]. A solid tile samples exactly; this only absorbs
/// rounding.
const TOLERANCE: u8 = 8;
/// An image id no image gets (they count up from zero); the probe's atlas is its own anyway.
const PROBE_IMAGE: ImageId = ImageId(usize::MAX);

/// The GPU a verdict was measured on.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Gpu {
    /// The adapter's LUID, low and high part.
    adapter: (u32, i32),
    /// The user-mode driver version.
    driver: i64,
}

/// Whether polychrome sprites draw through the unwrapped fragment shader, and the GPU that holds
/// for (`None` when it could not be read).
static VERDICT: Mutex<Option<(Option<Gpu>, bool)>> = Mutex::new(None);

/// Gives `renderer`'s polychrome pipeline the fragment shader its GPU draws photos with, probing
/// when there is no verdict for that GPU yet. Called whenever the pipelines are created.
pub(crate) fn apply(renderer: &mut DirectXRenderer) {
    let gpu = renderer
        .devices
        .as_ref()
        .and_then(|devices| gpu_of(&devices.adapter).log_err());
    let known = VERDICT
        .lock()
        .filter(|(measured_on, _)| *measured_on == gpu)
        .map(|(_, unclipped)| unclipped);
    match known {
        Some(true) => {
            unclip(renderer)
                .context("unclipping polychrome sprites")
                .log_err();
        }
        Some(false) => {}
        // `decide` leaves the pipeline with the shader it decided on.
        None => {
            if let Some(unclipped) = decide(renderer) {
                *VERDICT.lock() = Some((gpu, unclipped));
            }
        }
    }
}

fn gpu_of(adapter: &IDXGIAdapter1) -> Result<Gpu> {
    let luid = unsafe { adapter.GetDesc1() }?.AdapterLuid;
    let driver = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }?;
    Ok(Gpu {
        adapter: (luid.LowPart, luid.HighPart),
        driver,
    })
}

/// Whether polychrome sprites should draw through the unwrapped fragment shader, or `None` when the
/// probe could not run. Leaves `renderer`'s pipeline with the shader decided on.
fn decide(renderer: &mut DirectXRenderer) -> Option<bool> {
    match std::env::var(OVERRIDE_ENV).as_deref() {
        Ok("0") => {
            log::info!("{OVERRIDE_ENV}=0: photos are drawn without the rounded clip");
            return Some(
                unclip(renderer)
                    .context("unclipping polychrome sprites")
                    .log_err()
                    .is_some(),
            );
        }
        Ok("1") => return Some(false),
        _ => {}
    }
    let started = Instant::now();
    let wrapped = match probe(renderer) {
        Ok(pixels) if pixels.drawn() => {
            log::info!(
                "polychrome clip probe passed in {:?}: photos keep the rounded clip",
                started.elapsed()
            );
            return Some(false);
        }
        Ok(pixels) => pixels,
        Err(error) => {
            log::error!("polychrome clip probe failed to run: {error:#}");
            return None;
        }
    };
    let shader = renderer.pipelines.poly_sprites.fragment.clone();
    let unwrapped = unclip(renderer).and_then(|()| probe(renderer));
    let gpu = gpu_description(renderer);
    match unwrapped {
        Ok(pixels) if pixels.drawn() => {
            log::warn!(
                "photos are drawn without the rounded clip on {gpu}: \
                 the clipped shader drew {wrapped:?}, the unclipped one {pixels:?}"
            );
            Some(true)
        }
        outcome => {
            renderer.pipelines.poly_sprites.fragment = shader;
            match outcome {
                Ok(pixels) => {
                    log::error!(
                        "photos may not draw on {gpu}: \
                         the clipped shader drew {wrapped:?}, the unclipped one {pixels:?}"
                    );
                    Some(false)
                }
                Err(error) => {
                    log::error!(
                        "polychrome clip probe failed to run the unclipped shader on {gpu} \
                         (the clipped one drew {wrapped:?}): {error:#}"
                    );
                    None
                }
            }
        }
    }
}

/// Swaps the unwrapped fragment shader into `renderer`'s polychrome pipeline.
fn unclip(renderer: &mut DirectXRenderer) -> Result<()> {
    let device = &renderer.devices.as_ref().context("devices missing")?.device;
    renderer.pipelines.poly_sprites.fragment = unclipped_shader(device)?;
    Ok(())
}

fn unclipped_shader(device: &ID3D11Device) -> Result<ID3D11PixelShader> {
    let bytes = RawShaderBytes::new(
        ShaderModule::PolychromeSpriteUnclipped,
        ShaderTarget::Fragment,
    )?;
    create_fragment_shader(device, bytes.as_bytes())
}

/// The centre pixel of the probe sprite, RGBA, as drawn.
#[derive(Debug)]
struct ProbePixels {
    /// With no clip bound.
    plain: [u8; 4],
    /// Inside [`NESTED_CLIPS`] rounded clips.
    clipped: [u8; 4],
}

impl ProbePixels {
    fn drawn(&self) -> bool {
        [self.plain, self.clipped].iter().all(|pixel| {
            pixel
                .iter()
                .zip(COLOR)
                .all(|(&channel, expected)| channel.abs_diff(expected) <= TOLERANCE)
        })
    }
}

/// [`draw_probe`] through `renderer`'s pipelines.
fn probe(renderer: &mut DirectXRenderer) -> Result<ProbePixels> {
    let DirectXRenderer {
        devices,
        globals,
        pipelines,
        ..
    } = renderer;
    let devices = devices.as_ref().context("devices missing")?;
    let clip_side = TARGET_SIZE as f32;
    draw_probe(
        &devices.device,
        &devices.device_context,
        globals,
        pipelines,
        clip_side,
    )
}

/// Draws the probe sprite through `pipelines`' polychrome pipeline, plain and inside nested
/// rounded clips spanning `clip_side` × `clip_side` from the target's origin, into a texture of
/// its own, with an atlas of its own, and reads its centre back.
fn draw_probe(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    globals: &DirectXGlobalElements,
    pipelines: &mut DirectXRenderPipelines,
    clip_side: f32,
) -> Result<ProbePixels> {
    let atlas = DirectXAtlas::new(device, context);
    let image = COLOR.repeat((IMAGE_SIZE * IMAGE_SIZE) as usize);
    let key = AtlasKey::Image(RenderImageParams {
        image_id: PROBE_IMAGE,
        frame_index: 0,
        hdr: false,
    });
    let tile = atlas
        .get_or_insert_with(key, &mut || {
            let extent = size(DevicePixels(IMAGE_SIZE), DevicePixels(IMAGE_SIZE));
            Ok(Some((extent, Cow::Borrowed(image.as_slice()))))
        })?
        .context("probe image tile")?;
    let side = TARGET_SIZE as f32;
    let mut scene = Scene::default();
    scene.insert_primitive(PolychromeSprite {
        order: 0,
        pixelated: 0,
        transformation: TransformationMatrix::unit(),
        grayscale: false.into(),
        opacity: 1.,
        bounds: square(SPRITE_INSET, side - 2. * SPRITE_INSET),
        content_mask: ContentMask {
            bounds: square(0., side),
        },
        corner_radii: Corners::default(),
        tile,
    });
    scene.finish();

    let texture = create_target(device)?;
    let mut view = None;
    unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut view))? };
    let frame_globals =
        crate::fast::layers::tile_cache::create_globals(device, TARGET_SIZE, TARGET_SIZE)?;
    let clips =
        crate::fast::rounded_clip::probe_view(device, clip_side, CLIP_RADIUS, NESTED_CLIPS)?;
    let target = Target {
        device,
        device_context: context,
        atlas: &atlas,
        globals,
        paths: None,
    };
    let mut frame = FrameState::default();
    let drawn = (|| -> Result<ProbePixels> {
        bind_target(context, &view, &frame_globals, globals)?;
        upload(&target, pipelines, &scene)?;
        draw_scene(&target, pipelines, &mut frame, &scene)?;
        let plain = centre(device, context, &texture)?;
        bind_target(context, &view, &frame_globals, globals)?;
        draw_scene_masked(&target, pipelines, &mut frame, &scene, clips.as_ref())?;
        let clipped = centre(device, context, &texture)?;
        Ok(ProbePixels { plain, clipped })
    })();
    // Whether or not the draws got through, nothing of the probe's stays bound on the shared
    // context: not its clip, which the wrapped shaders would apply to whatever draws next, and not
    // its target.
    let unbound = crate::fast::rounded_clip::bind(&target, &mut frame.rounded_masks, None);
    unsafe { context.OMSetRenderTargets(None, None) };
    drawn.and_then(|pixels| unbound.map(|()| pixels))
}

/// Clears `view` and binds it as `DirectXRenderer::pre_draw` binds the window's target.
fn bind_target(
    context: &ID3D11DeviceContext,
    view: &Option<ID3D11RenderTargetView>,
    frame_globals: &Option<ID3D11Buffer>,
    globals: &DirectXGlobalElements,
) -> Result<()> {
    let side = TARGET_SIZE as f32;
    unsafe {
        context.ClearRenderTargetView(view.as_ref().context("probe target view")?, &[0.; 4]);
        context.OMSetRenderTargets(Some(slice::from_ref(view)), None);
        context.RSSetViewports(Some(&[D3D11_VIEWPORT {
            TopLeftX: 0.,
            TopLeftY: 0.,
            Width: side,
            Height: side,
            MinDepth: 0.,
            MaxDepth: 1.,
        }]));
        context.VSSetConstantBuffers(0, Some(slice::from_ref(frame_globals)));
        context.VSSetConstantBuffers(1, Some(slice::from_ref(&globals.batch_params_buffer)));
        context.PSSetConstantBuffers(0, Some(slice::from_ref(frame_globals)));
    }
    Ok(())
}

/// A [`TARGET_SIZE`] square target of the window's format.
fn create_target(device: &ID3D11Device) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: TARGET_SIZE,
        Height: TARGET_SIZE,
        MipLevels: 1,
        ArraySize: 1,
        Format: RENDER_TARGET_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    texture.context("probe target")
}

/// The centre pixel of `texture`, a [`TARGET_SIZE`] BGRA target, as RGBA.
fn centre(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    texture: &ID3D11Texture2D,
) -> Result<[u8; 4]> {
    let readback = read_back_texture(device, context, texture)?;
    let middle = TARGET_SIZE as usize / 2;
    let at = (middle * readback.width as usize + middle) * 4;
    let [b, g, r, a]: [u8; 4] = readback
        .bytes
        .get(at..at + 4)
        .context("probe readback")?
        .try_into()?;
    Ok([r, g, b, a])
}

fn square(origin: f32, side: f32) -> Bounds<ScaledPixels> {
    Bounds {
        origin: point(ScaledPixels(origin), ScaledPixels(origin)),
        size: size(ScaledPixels(side), ScaledPixels(side)),
    }
}

fn gpu_description(renderer: &DirectXRenderer) -> String {
    match renderer.gpu_specs() {
        Ok(gpu) => format!(
            "{} ({}, driver {})",
            gpu.device_name, gpu.driver_name, gpu.driver_info
        ),
        Err(error) => format!("an unknown GPU ({error})"),
    }
}

#[cfg(test)]
mod tests {
    use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};

    use super::*;
    use crate::fast::layers::tests::{create_device, set_rasterizer_state};

    /// What the probe draws with on each Direct3D 11 device here: the hardware one and WARP (the
    /// only one on CI).
    fn devices() -> Vec<(
        ID3D11Device,
        ID3D11DeviceContext,
        DirectXGlobalElements,
        DirectXRenderPipelines,
    )> {
        [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP]
            .into_iter()
            .filter_map(create_device)
            .map(|(device, context)| {
                set_rasterizer_state(&device, &context).unwrap();
                let globals = DirectXGlobalElements::new(&device).unwrap();
                let pipelines = DirectXRenderPipelines::new(&device).unwrap();
                (device, context, globals, pipelines)
            })
            .collect()
    }

    /// A GPU that draws both shaders right must pass, or every photo on it loses its rounded
    /// corners for nothing.
    #[test]
    fn the_probe_sees_its_sprite_through_both_shaders() {
        let side = TARGET_SIZE as f32;
        for (device, context, globals, mut pipelines) in devices() {
            let wrapped = draw_probe(&device, &context, &globals, &mut pipelines, side).unwrap();
            assert!(wrapped.drawn(), "clipped shader: {wrapped:?}");

            pipelines.poly_sprites.fragment = unclipped_shader(&device).unwrap();
            let unwrapped = draw_probe(&device, &context, &globals, &mut pipelines, side).unwrap();
            assert!(unwrapped.drawn(), "unclipped shader: {unwrapped:?}");
        }
    }

    /// The clipped draw really goes through the rounded clip: clips that stop short of the
    /// sprite's centre leave it undrawn there with the clipped shader, and not with the unclipped
    /// one, which ignores them.
    #[test]
    fn the_clipped_draw_is_clipped() {
        let short = SPRITE_INSET * 2.;
        for (device, context, globals, mut pipelines) in devices() {
            let wrapped = draw_probe(&device, &context, &globals, &mut pipelines, short).unwrap();
            assert_eq!(wrapped.clipped, [0; 4], "{wrapped:?}");
            assert!(!wrapped.drawn());

            pipelines.poly_sprites.fragment = unclipped_shader(&device).unwrap();
            let unwrapped = draw_probe(&device, &context, &globals, &mut pipelines, short).unwrap();
            assert!(unwrapped.drawn(), "{unwrapped:?}");
        }
    }

    /// A sprite that did not draw leaves the cleared target: the probe must see that as missing.
    #[test]
    fn a_cleared_pixel_is_not_drawn() {
        let missing = ProbePixels {
            plain: COLOR,
            clipped: [0; 4],
        };
        assert!(!missing.drawn());
    }
}

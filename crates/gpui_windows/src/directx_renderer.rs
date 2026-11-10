use std::{
    slice,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
use windows::{
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
        },
    },
    core::{HSTRING, Interface},
};

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::hdr_display::{self, HdrDisplay};
use crate::*;
use gpui::*;

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";
pub(crate) const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
/// Swap chain and scene targets while HDR output is on (see [`HdrOutput`]). The swap chain is then
/// scRGB — linear light, BT.709 primaries, 1.0 = 80 nits — and the scene targets keep the same
/// sRGB-encoded values as the 8-bit pipeline, only with room above 1.0.
const HDR_RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;
/// How often the display under the window is re-read while HDR is wanted — the app asks every
/// frame, but monitor moves, the HDR toggle and the SDR brightness slider send no message to the
/// renderer.
const HDR_PROBE_INTERVAL: Duration = Duration::from_secs(1);
/// Set to log statistics of the HDR swap chain (scRGB min/max/mean, share above SDR white) at most
/// every [`HDR_DIAGNOSTICS_INTERVAL`]. HDR can't be checked by eye or screenshot — capture APIs
/// tone-map it — so this is how its output gets verified.
const HDR_DIAGNOSTICS_ENV: &str = "GPUI_HDR_DIAGNOSTICS";
const HDR_DIAGNOSTICS_INTERVAL: Duration = Duration::from_secs(2);
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Number of content-filter (`filter`) nesting levels that get their own isolated group target.
/// Two covers the realistic "a blurred element inside another blurred element" case; deeper nests
/// render inline (unblurred at the inner level) rather than allocating unbounded VRAM. Must match
/// the wgpu backend's `MAX_FILTER_DEPTH` so nested blur renders consistently across platforms.
const MAX_FILTER_DEPTH: usize = 2;

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    pub(crate) atlas: Arc<DirectXAtlas>,
    pub(crate) devices: Option<DirectXRendererDevices>,
    pub(crate) resources: Option<DirectXResources>,
    pub(crate) globals: DirectXGlobalElements,
    pub(crate) pipelines: DirectXRenderPipelines,
    pub(crate) direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,

    pub(crate) width: u32,
    pub(crate) height: u32,

    /// Whether we want to skip drawing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    pub(crate) skip_draws: bool,
    pub(crate) fast_frame: crate::fast::frame::FrameState,
    pub(crate) fast_composition: crate::fast::composition::DirectXComposition,
    pub(crate) fast_recovery: crate::fast::device_recovery::RecoveryState,

    /// The render target currently bound for the main scene this frame (the offscreen
    /// `scene_color` when blur filters are present, a content-filter group texture inside such a
    /// group, or the swapchain otherwise). `draw_paths_to_intermediate` restores to this after
    /// its own pass so paths land on the correct target.
    active_render_target: Option<ID3D11RenderTargetView>,

    hdr: HdrOutput,
}

/// HDR (scRGB) output. Off until the app asks for it ([`DirectXRenderer::set_hdr_content`]) *and*
/// the window's display has Windows HDR on; only then are the swap chain and the scene targets
/// FP16, so every other window keeps the 8-bit pipeline and half the memory.
///
/// While on, the scene is drawn into the FP16 `scene_color` exactly as it would be drawn into the
/// 8-bit swap chain — sRGB-encoded, premultiplied — so blending, text gamma and blur stay as tuned;
/// a final pass converts it to scRGB, putting SDR white at the user's SDR brightness.
#[derive(Default)]
struct HdrOutput {
    /// What the app asked for last.
    wanted: bool,
    /// The display's current headroom (multiples of SDR white) that HDR sprites are soft-clipped
    /// into, as the app reported it.
    headroom: f32,
    /// The swap chain and scene targets are FP16 right now.
    active: bool,
    /// Switching to FP16 failed; stay SDR until the app renews its request.
    failed: bool,
    /// The display under the window, re-read every [`HDR_PROBE_INTERVAL`] while HDR is wanted.
    display: Option<HdrDisplay>,
    last_probe: Option<Instant>,
    /// Factory used only for probing, replaced whenever DXGI reports it stale.
    probe_factory: Option<IDXGIFactory1>,
    last_diagnostics: Option<Instant>,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    dxgi_device: Option<IDXGIDevice>,
    pub(crate) annotation: Option<ID3DUserDefinedAnnotation>,
}

pub(crate) struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    pub(crate) render_target_view: Option<ID3D11RenderTargetView>,

    // Path intermediate textures (with MSAA)
    pub(crate) path_intermediate_texture: ID3D11Texture2D,
    pub(crate) path_intermediate_srv: Option<ID3D11ShaderResourceView>,
    pub(crate) path_intermediate_msaa_texture: ID3D11Texture2D,
    pub(crate) path_intermediate_msaa_view: Option<ID3D11RenderTargetView>,

    // Offscreen targets for blur filters (each is render-target + shader-resource).
    pub(crate) blur: BlurResources,

    // Cached viewport
    pub(crate) viewport: D3D11_VIEWPORT,
}

/// Offscreen render targets used by the blur filters. The scene is rendered into `scene_color`
/// (so filters can sample it), `ping`/`pong` are half-resolution scratch for the separable
/// gaussian, and `groups` isolate content-filter (`filter`) subtrees — one per nesting level
/// (indexed by isolation depth), up to [`MAX_FILTER_DEPTH`], so nested content blurs isolate
/// correctly; deeper nests render inline.
pub(crate) struct BlurResources {
    #[expect(dead_code)]
    scene_color: ID3D11Texture2D,
    pub(crate) scene_color_rtv: Option<ID3D11RenderTargetView>,
    pub(crate) scene_color_srv: Option<ID3D11ShaderResourceView>,
    #[expect(dead_code)]
    ping: ID3D11Texture2D,
    pub(crate) ping_rtv: Option<ID3D11RenderTargetView>,
    pub(crate) ping_srv: Option<ID3D11ShaderResourceView>,
    #[expect(dead_code)]
    pong: ID3D11Texture2D,
    pub(crate) pong_rtv: Option<ID3D11RenderTargetView>,
    pub(crate) pong_srv: Option<ID3D11ShaderResourceView>,
    // Kept alive for the lifetime of their views; indexed by isolation depth.
    #[expect(dead_code)]
    groups: Vec<ID3D11Texture2D>,
    pub(crate) group_rtvs: Vec<Option<ID3D11RenderTargetView>>,
    pub(crate) group_srvs: Vec<Option<ID3D11ShaderResourceView>>,
}

impl BlurResources {
    /// `format` follows the swap chain: FP16 while HDR output is on, so a blurred or isolated
    /// group over an HDR image keeps its highlights.
    fn new(device: &ID3D11Device, width: u32, height: u32, format: DXGI_FORMAT) -> Result<Self> {
        let half_w = (width / 2).max(1);
        let half_h = (height / 2).max(1);
        let (scene_color, scene_color_rtv, scene_color_srv) =
            create_color_target(device, width, height, format)?;
        let (ping, ping_rtv, ping_srv) = create_color_target(device, half_w, half_h, format)?;
        let (pong, pong_rtv, pong_srv) = create_color_target(device, half_w, half_h, format)?;
        let mut groups = Vec::with_capacity(MAX_FILTER_DEPTH);
        let mut group_rtvs = Vec::with_capacity(MAX_FILTER_DEPTH);
        let mut group_srvs = Vec::with_capacity(MAX_FILTER_DEPTH);
        for _ in 0..MAX_FILTER_DEPTH {
            let (group, group_rtv, group_srv) = create_color_target(device, width, height, format)?;
            groups.push(group);
            group_rtvs.push(group_rtv);
            group_srvs.push(group_srv);
        }
        Ok(Self {
            scene_color,
            scene_color_rtv,
            scene_color_srv,
            ping,
            ping_rtv,
            ping_srv,
            pong,
            pong_rtv,
            pong_srv,
            groups,
            group_rtvs,
            group_srvs,
        })
    }
}

pub(crate) struct DirectXRenderPipelines {
    pub(crate) shadow_pipeline: PipelineState<Shadow>,
    pub(crate) quad_pipeline: PipelineState<Quad>,
    pub(crate) path_rasterization_pipeline: PipelineState<PathRasterizationSprite>,
    pub(crate) path_sprite_pipeline: PipelineState<PathSprite>,
    pub(crate) underline_pipeline: PipelineState<Underline>,
    pub(crate) mono_sprites: PipelineState<MonochromeSprite>,
    pub(crate) subpixel_sprites: PipelineState<SubpixelSprite>,
    pub(crate) poly_sprites: PipelineState<PolychromeSprite>,
    // Blur (backdrop-filter / filter). These don't use the generic PipelineState since they
    // sample a texture rather than read a structured instance buffer; their parameters live in
    // a dedicated constant buffer at register b3.
    pub(crate) blur_downsample_vertex: ID3D11VertexShader,
    pub(crate) blur_downsample_fragment: ID3D11PixelShader,
    pub(crate) blur_vertex: ID3D11VertexShader,
    pub(crate) blur_fragment: ID3D11PixelShader,
    pub(crate) blur_composite_vertex: ID3D11VertexShader,
    pub(crate) blur_composite_fragment: ID3D11PixelShader,
    pub(crate) blur_params_buffer: ID3D11Buffer,
    pub(crate) blur_blend_replace: ID3D11BlendState,
    pub(crate) blur_blend_composite: ID3D11BlendState,
    // The final scRGB pass while HDR output is on (see `HdrOutput`); same full-screen draw as the
    // scene blit.
    pub(crate) hdr_output_vertex: ID3D11VertexShader,
    pub(crate) hdr_output_fragment: ID3D11PixelShader,
}

pub(crate) struct DirectXGlobalElements {
    pub(crate) global_params_buffer: Option<ID3D11Buffer>,
    pub(crate) batch_params_buffer: Option<ID3D11Buffer>,
    pub(crate) sampler: Option<ID3D11SamplerState>,
}

struct Annotation<'a>(&'a ID3DUserDefinedAnnotation);

impl<'a> Annotation<'a> {
    fn new(annotation: &'a ID3DUserDefinedAnnotation, label: HSTRING) -> Self {
        unsafe { annotation.BeginEvent(&label) };
        Self(annotation)
    }
}

impl Drop for Annotation<'_> {
    fn drop(&mut self) {
        unsafe { self.0.EndEvent() };
    }
}

pub(crate) struct DirectComposition {
    pub(crate) comp_device: IDCompositionDevice,
    pub(crate) comp_target: IDCompositionTarget,
    pub(crate) comp_visual: IDCompositionVisual,
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };
        let annotation = device_context.cast().ok();

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            dxgi_device,
            annotation,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            direct_composition,
            font_info: Self::get_font_info(),
            width: 1,
            height: 1,
            skip_draws: false,
            fast_frame: crate::fast::frame::FrameState::default(),
            fast_composition: crate::fast::composition::DirectXComposition::default(),
            fast_recovery: crate::fast::device_recovery::RecoveryState::default(),
            active_render_target: None,
            hdr: HdrOutput::default(),
        })
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(&self, clear_color: &[f32; 4]) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let device_context = &self
            .devices
            .as_ref()
            .expect("devices missing")
            .device_context;
        crate::fast::globals::write_globals(
            &self.fast_frame,
            device_context,
            self.globals.global_params_buffer.as_ref().unwrap(),
            &[GlobalParams {
                gamma_ratios: self.font_info.gamma_ratios,
                viewport_size: [resources.viewport.Width, resources.viewport.Height],
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
                is_bgr: self.font_info.is_bgr as u32,
                // An SDR target has no headroom: HDR sprites are compressed into SDR.
                hdr_headroom: if self.hdr.active {
                    self.hdr.headroom
                } else {
                    1.0
                },
                hdr_output_scale: self.hdr_output_scale(),
                _pad: 0,
            }],
        )?;
        unsafe {
            device_context.ClearRenderTargetView(
                resources
                    .render_target_view
                    .as_ref()
                    .context("missing render target view")?,
                clear_color,
            );
            device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
            device_context
                .VSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            device_context
                .VSSetConstantBuffers(1, Some(slice::from_ref(&self.globals.batch_params_buffer)));
            device_context
                .PSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn present(&mut self) -> Result<()> {
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
        };
        #[cfg(test)]
        crate::fast::native_tests::present(self.hwnd, self.direct_composition.is_some(), result.0);
        result.ok().context("Presenting swap chain failed")
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        crate::fast::layers::release_tiles(self);
        crate::fast::composition::release(self);
        let disable_direct_composition = self.direct_composition.is_none();

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.globals = globals;
        self.pipelines = pipelines;
        self.direct_composition = direct_composition;
        crate::fast::composition::recreate(self)?;
        self.skip_draws = true;
        // The new swap chain is 8-bit; forgetting the request makes the app's next one re-probe
        // the display and switch HDR back on.
        self.hdr = HdrOutput {
            headroom: self.hdr.headroom,
            ..HdrOutput::default()
        };
        Ok(())
    }

    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if self.skip_draws || !crate::fast::device_recovery::can_draw(self) {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            return Ok(());
        }
        self.render(scene, background_appearance)?;
        if self.hdr.active {
            self.log_hdr_diagnostics();
        }
        self.present()
    }

    /// Clear the render target for `background_appearance` and encode every
    /// primitive batch of `scene` into it, without presenting. Shared by
    /// [`draw`](Self::draw) (which then presents) and
    /// [`render_to_image`](Self::render_to_image) (which reads the target back
    /// instead), so the two cannot drift.
    pub(crate) fn render(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        crate::fast::layers::raster::rasterize_tiles(self, scene)?;
        let clear_color = match background_appearance {
            WindowBackgroundAppearance::Opaque => [1.0f32; 4],
            _ => [0.0f32; 4],
        };
        self.pre_draw(&clear_color)?;
        self.upload_scene_buffers(scene)?;

        // [PiCell] Route through the offscreen scene texture when the scene contains blur
        // filters, or always while HDR output is on (its final pass reads the scene from there);
        // otherwise render straight to the swapchain exactly as before.
        let filters =
            crate::picell_filters::FilterTargets::begin(self, scene, self.hdr.active, clear_color)?;
        self.fast_frame.picell_filters = filters;
        let drawn = crate::fast::frame::draw_batches(self, scene);
        let mut filters = self.fast_frame.picell_filters.take();
        if !drawn? {
            self.draw_batches_labeled(scene, &mut filters)?;
        }
        if let Some(filters) = filters {
            filters.finish()?;
        }
        Ok(())
    }

    /// Upstream's batch loop, each batch inside a labeled event: what runs while a graphics
    /// debugger such as PIX captures the frame, in place of gpui-fast's
    /// (`fast::frame::draw_batches`).
    fn draw_batches_labeled(
        &mut self,
        scene: &Scene,
        filters: &mut Option<crate::picell_filters::FilterTargets>,
    ) -> Result<()> {
        // Paths restore to the target the batches draw into (see `draw_paths_to_intermediate`).
        self.active_render_target = filters.as_ref().and_then(|f| f.current_rtv.clone());
        let annotation = self
            .devices
            .as_ref()
            .and_then(|devices| devices.annotation.clone())
            .filter(|annotation| unsafe { annotation.GetStatus().as_bool() });
        for batch in scene.batches() {
            let _annotation = annotation
                .as_ref()
                .map(|annotation| Annotation::new(annotation, HSTRING::from(batch.label())));
            match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(range.start, range.len()),
                PrimitiveBatch::Quads(range) => self.draw_quads(range.start, range.len()),
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    self.draw_paths_to_intermediate(paths)?;
                    self.draw_paths_from_intermediate(paths)
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(range.start, range.len()),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_monochrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    self.draw_subpixel_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_polychrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(&scene.surfaces[range]),
                PrimitiveBatch::BackdropFilters(range) => match filters.as_mut() {
                    Some(filters) => filters.backdrop_filters(&scene.backdrop_filters[range]),
                    None => Ok(()),
                },
                PrimitiveBatch::FilterBoundary(ix) => match filters.as_mut() {
                    Some(filters) => {
                        let result = filters.boundary(&scene.filter_boundaries[ix]);
                        self.active_render_target = filters.current_rtv.clone();
                        result
                    }
                    None => Ok(()),
                },
            }
            .with_context(|| {
                format!(
                    "scene too large:\
                    {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                )
            })?;
        }
        self.active_render_target = None;
        Ok(())
    }

    /// Render `scene` to an offscreen CPU image **without presenting** so
    /// the window need never be shown or visible (the macOS headless path
    /// goes through MetalRenderer; this is the Windows analogue). Draws into
    /// the existing render target, reads it back, and converts BGRA to RGBA —
    /// or, while HDR output is on, undoes the scRGB pass so the image holds
    /// what the 8-bit pipeline would (HDR highlights clipped to white).
    #[cfg(feature = "test-support")]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        // A pending device-lost recovery (`skip_draws`) leaves the atlas holding
        // tile references from the previous device; drawing before the forced
        // re-render rebuilds them panics in `DirectXAtlasTextures::texture`.
        anyhow::ensure!(
            !self.skip_draws,
            "render_to_image unavailable while recovering from a lost device"
        );
        self.render(scene, background_appearance)?;

        let readback = self.read_back_render_target()?;
        let (width, height) = (readback.width, readback.height);
        let pixels = if self.hdr.active {
            let scale = self.hdr_output_scale();
            readback
                .rgba_f32()
                .chunks_exact(4)
                .flat_map(|px| scrgb_to_srgb8(px, scale))
                .collect()
        } else {
            // The render target is BGRA; image::RgbaImage expects RGBA.
            let mut pixels = readback.bytes;
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
            pixels
        };
        image::RgbaImage::from_raw(width, height, pixels)
            .context("Failed to build RgbaImage from staging readback")
    }

    /// A CPU copy of the swapchain's current back buffer.
    fn read_back_render_target(&self) -> Result<TextureReadback> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let render_target = self
            .resources
            .as_ref()
            .context("resources missing")?
            .render_target
            .as_ref()
            .context("render target missing")?;
        read_back_texture(&devices.device, &devices.device_context, render_target)
    }

    /// The scRGB value SDR white is drawn at while HDR output is on.
    fn hdr_output_scale(&self) -> f32 {
        self.hdr
            .display
            .map_or(1.0, |display| display.sdr_white_scale())
    }

    /// Ask for HDR output (`enabled`) and report the display's current `headroom` (multiples of
    /// SDR white) that HDR sprites are soft-clipped into. Called after every drawn frame, so it
    /// mostly just stores the request; the display is re-read at most every
    /// [`HDR_PROBE_INTERVAL`], and the swap chain is switched only when the outcome changes.
    pub(crate) fn set_hdr_content(&mut self, enabled: bool, headroom: f32) {
        self.hdr.headroom = headroom.max(1.0);
        if enabled && !self.hdr.wanted {
            // A fresh request: look at the display now and give a failed switch another chance.
            self.hdr.failed = false;
            self.hdr.last_probe = None;
        }
        self.hdr.wanted = enabled;
        if enabled
            && self
                .hdr
                .last_probe
                .is_none_or(|probed| probed.elapsed() >= HDR_PROBE_INTERVAL)
        {
            self.hdr.display = self.probe_hdr_display();
            self.hdr.last_probe = Some(Instant::now());
        }

        let active = enabled
            && !self.hdr.failed
            && self.hdr.display.is_some_and(|display| display.hdr_enabled);
        if active == self.hdr.active {
            return;
        }
        match self.switch_hdr_output(active) {
            Ok(()) => log::info!(
                "HDR output {}: {:?}, scRGB scale {:.3}",
                if active { "on" } else { "off" },
                self.hdr.display,
                self.hdr_output_scale()
            ),
            Err(err) => {
                log::warn!("Switching HDR output failed, staying SDR: {err:#}");
                self.hdr.failed = true;
            }
        }
    }

    fn probe_hdr_display(&mut self) -> Option<HdrDisplay> {
        let stale = self
            .hdr
            .probe_factory
            .as_ref()
            .is_none_or(|factory| !unsafe { factory.IsCurrent() }.as_bool());
        if stale {
            self.hdr.probe_factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }
                .context("Creating DXGI factory for the HDR display probe")
                .log_err();
        }
        hdr_display::probe(self.hdr.probe_factory.as_ref()?, self.hwnd)
    }

    /// Rebuild the swap chain buffers and scene targets in the format for `hdr`. A failed switch
    /// to HDR still leaves a working 8-bit swap chain behind — it can fail after `ResizeBuffers`
    /// already made the buffers FP16.
    fn switch_hdr_output(&mut self, hdr: bool) -> Result<()> {
        self.hdr.active = hdr;
        let result = self.rebuild_swap_chain_buffers().and_then(|()| {
            if hdr {
                return Ok(());
            }
            // Back to the 8-bit swapchain's default colour space.
            let resources = self.resources.as_ref().context("resources missing")?;
            set_swap_chain_color_space(
                &resources.swap_chain,
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
            )
        });
        if result.is_err() && hdr {
            self.hdr.active = false;
            self.rebuild_swap_chain_buffers()
                .context("Restoring the SDR swap chain")
                .log_err();
        }
        result
    }

    /// Statistics of the HDR swapchain, logged when [`HDR_DIAGNOSTICS_ENV`] is set.
    fn log_hdr_diagnostics(&mut self) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var_os(HDR_DIAGNOSTICS_ENV).is_some())
            || self
                .hdr
                .last_diagnostics
                .is_some_and(|logged| logged.elapsed() < HDR_DIAGNOSTICS_INTERVAL)
        {
            return;
        }
        self.hdr.last_diagnostics = Some(Instant::now());
        let Some(readback) = self.read_back_render_target().log_err() else {
            return;
        };
        let scale = self.hdr_output_scale();
        let rgba = readback.rgba_f32();
        let (mut min, mut max, mut sum, mut above_white) = (f32::MAX, f32::MIN, 0.0f64, 0usize);
        for px in rgba.chunks_exact(4) {
            let peak = px[0].max(px[1]).max(px[2]);
            min = min.min(px[0].min(px[1]).min(px[2]));
            max = max.max(peak);
            sum += (px[0] + px[1] + px[2]) as f64 / 3.0;
            // A hair over SDR white, so rounding of UI white doesn't count.
            if peak > scale * 1.01 {
                above_white += 1;
            }
        }
        let pixel_count = (rgba.len() / 4).max(1);
        log::info!(
            "HDR diagnostics {}x{}: scRGB min {min:.4} max {max:.4} mean {:.4}; SDR white {scale:.3} \
             ({:.0} nits), {:.2}% of pixels brighter, headroom {:.2}",
            readback.width,
            readback.height,
            sum / pixel_count as f64,
            scale * hdr_display::SCRGB_WHITE_NITS,
            above_white as f64 * 100.0 / pixel_count as f64,
            self.hdr.headroom,
        );
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        crate::fast::layers::release_tiles(self);
        self.rebuild_swap_chain_buffers()
    }

    /// Recreate the swap chain buffers and every size- or format-dependent target for the current
    /// size and HDR state. `ResizeBuffers` takes the format too, so switching HDR needs no new
    /// swap chain.
    fn rebuild_swap_chain_buffers(&mut self) -> Result<()> {
        let (width, height) = (self.width, self.height);
        let format = if self.hdr.active {
            HDR_RENDER_TARGET_FORMAT
        } else {
            RENDER_TARGET_FORMAT
        };

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    format,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }
        if self.hdr.active {
            // Re-applied after every resize — the buffers are new.
            set_swap_chain_color_space(
                &resources.swap_chain,
                DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
            )?;
        }

        resources.recreate_resources(devices, width, height, format)?;

        unsafe {
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(&resources.render_target_view)), None);
        }
        crate::fast::composition::resize(self)?;

        Ok(())
    }

    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;

        if !scene.shadows.is_empty() {
            self.pipelines.shadow_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.shadows,
            )?;
        }

        if !scene.quads.is_empty() {
            self.pipelines.quad_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.quads,
            )?;
        }

        if !scene.underlines.is_empty() {
            self.pipelines.underline_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.underlines,
            )?;
        }

        if !scene.monochrome_sprites.is_empty() {
            self.pipelines.mono_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.monochrome_sprites,
            )?;
        }

        if !scene.subpixel_sprites.is_empty() {
            self.pipelines.subpixel_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.subpixel_sprites,
            )?;
        }

        if !scene.polychrome_sprites.is_empty() {
            self.pipelines.poly_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.polychrome_sprites,
            )?;
        }

        Ok(())
    }

    fn draw_shadows(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.shadow_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_quads(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.quad_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_paths_to_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        // Clear intermediate MSAA texture
        unsafe {
            devices.device_context.ClearRenderTargetView(
                resources.path_intermediate_msaa_view.as_ref().unwrap(),
                &[0.0; 4],
            );
            // Set intermediate MSAA texture as render target
            devices.device_context.OMSetRenderTargets(
                Some(slice::from_ref(&resources.path_intermediate_msaa_view)),
                None,
            );
        }

        // Collect all vertices and sprites for a single draw call
        let mut vertices = Vec::new();

        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationSprite {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
            }));
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &vertices,
        )?;

        self.pipelines.path_rasterization_pipeline.draw(
            &devices.device_context,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            vertices.len() as u32,
            1,
        )?;

        // Resolve MSAA to non-MSAA intermediate texture
        unsafe {
            devices.device_context.ResolveSubresource(
                &resources.path_intermediate_texture,
                0,
                &resources.path_intermediate_msaa_texture,
                0,
                RENDER_TARGET_FORMAT,
            );
            // Restore the active render target (the offscreen scene/group target when blurring,
            // otherwise the swapchain) so the path sprites land on the correct surface.
            let restore_target = if self.active_render_target.is_some() {
                &self.active_render_target
            } else {
                &resources.render_target_view
            };
            devices
                .device_context
                .OMSetRenderTargets(Some(slice::from_ref(restore_target)), None);
        }

        Ok(())
    }

    fn draw_paths_from_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites = if paths.last().unwrap().order == first_path.order {
            paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect::<Vec<_>>()
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            &sprites,
        )?;

        // Draw the sprites with the path texture
        self.pipelines.path_sprite_pipeline.draw_with_texture(
            &devices.device_context,
            slice::from_ref(&resources.path_intermediate_srv),
            slice::from_ref(&self.globals.sampler),
            sprites.len() as u32,
        )
    }

    fn draw_underlines(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.underline_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            start as u32,
            len as u32,
        )
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let Some(texture_view) = self.atlas.get_texture_view(texture_id) else {
            return Ok(());
        };
        self.pipelines.mono_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let Some(texture_view) = self.atlas.get_texture_view(texture_id) else {
            return Ok(());
        };
        self.pipelines.subpixel_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let Some(texture_view) = self.atlas.get_texture_view(texture_id) else {
            return Ok(());
        };
        self.pipelines.poly_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_surfaces(&mut self, surfaces: &[PaintSurface]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        Ok(())
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
            let render_params: IDWriteRenderingParams1 =
                factory.CreateRenderingParams().unwrap().cast().unwrap();
            FontInfo {
                gamma_ratios: gpui::get_gamma_correction_ratios(render_params.GetGamma()),
                grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
                is_bgr: render_params.GetPixelGeometry() == DWRITE_PIXEL_GEOMETRY_BGR,
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
}

impl DirectXResources {
    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (
            render_target,
            render_target_view,
            path_intermediate_texture,
            path_intermediate_srv,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            viewport,
        ) = create_resources(devices, &swap_chain, width, height)?;
        set_rasterizer_state(&devices.device, &devices.device_context)?;
        let blur = BlurResources::new(&devices.device, width, height, RENDER_TARGET_FORMAT)?;

        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
            path_intermediate_texture,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            path_intermediate_srv,
            blur,
            viewport,
        })
    }

    /// `format` is the swap chain's, which the blur/scene targets follow.
    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<()> {
        let (
            render_target,
            render_target_view,
            path_intermediate_texture,
            path_intermediate_srv,
            path_intermediate_msaa_texture,
            path_intermediate_msaa_view,
            viewport,
        ) = create_resources(devices, &self.swap_chain, width, height)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        self.path_intermediate_texture = path_intermediate_texture;
        self.path_intermediate_msaa_texture = path_intermediate_msaa_texture;
        self.path_intermediate_msaa_view = path_intermediate_msaa_view;
        self.path_intermediate_srv = path_intermediate_srv;
        self.blur = BlurResources::new(&devices.device, width, height, format)?;
        self.viewport = viewport;
        Ok(())
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_blend_state_for_path_rasterization(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_blend_state_for_path_sprite(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?;

        let blur_downsample_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurDownsample, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_downsample_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurDownsample, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::Blur, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::Blur, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_composite_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurComposite, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let blur_composite_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::BlurComposite, ShaderTarget::Fragment)?.as_bytes(),
        )?;
        let blur_params_buffer =
            create_constant_buffer_sized(device, crate::picell_filters::BLUR_PARAMS_SIZE)?;
        let blur_blend_replace = create_blend_state_no_blend(device)?;
        // Premultiplied (One / InvSrcAlpha) — the composite outputs a premultiplied blurred sample;
        // straight-alpha blending would darken the faded edges.
        let blur_blend_composite = create_blend_state_for_path_sprite(device)?;
        let hdr_output_vertex = create_vertex_shader(
            device,
            RawShaderBytes::new(ShaderModule::HdrOutput, ShaderTarget::Vertex)?.as_bytes(),
        )?;
        let hdr_output_fragment = create_fragment_shader(
            device,
            RawShaderBytes::new(ShaderModule::HdrOutput, ShaderTarget::Fragment)?.as_bytes(),
        )?;

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
            blur_downsample_vertex,
            blur_downsample_fragment,
            blur_vertex,
            blur_fragment,
            blur_composite_vertex,
            blur_composite_fragment,
            blur_params_buffer,
            blur_blend_replace,
            blur_blend_composite,
            hdr_output_vertex,
            hdr_output_fragment,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, false) }?;
        let comp_visual = unsafe { comp_device.CreateVisual() }?;

        Ok(Self {
            comp_device,
            comp_target,
            comp_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.comp_visual.SetContent(swap_chain)?;
            self.comp_target.SetRoot(&self.comp_visual)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let global_params_buffer = create_constant_buffer::<GlobalParams>(device)?;
        let batch_params_buffer = create_constant_buffer::<BatchParams>(device)?;

        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        Ok(Self {
            global_params_buffer,
            batch_params_buffer,
            sampler,
        })
    }
}

#[derive(Debug, Default)]
#[repr(C)]
pub(crate) struct GlobalParams {
    pub(crate) gamma_ratios: [f32; 4],
    pub(crate) viewport_size: [f32; 2],
    pub(crate) grayscale_enhanced_contrast: f32,
    pub(crate) subpixel_enhanced_contrast: f32,
    pub(crate) is_bgr: u32,
    /// Display headroom (multiples of SDR white) HDR sprites are soft-clipped into; 1.0 on an SDR
    /// target.
    pub(crate) hdr_headroom: f32,
    /// scRGB value of SDR white for the HDR output pass.
    pub(crate) hdr_output_scale: f32,
    pub(crate) _pad: u32,
}

const _: () = assert!(std::mem::size_of::<GlobalParams>() == 48);

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
struct BatchParams {
    start_index: u32,
    _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<BatchParams>() == 16);

pub(crate) struct PipelineState<T> {
    pub(crate) label: &'static str,
    pub(crate) vertex: ID3D11VertexShader,
    pub(crate) fragment: ID3D11PixelShader,
    buffer: ID3D11Buffer,
    pub(crate) buffer_size: usize,
    pub(crate) view: Option<ID3D11ShaderResourceView>,
    pub(crate) blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            vertex,
            fragment,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    pub(crate) fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        if self.buffer_size < data.len() {
            let element_size = std::mem::size_of::<T>();
            let required_size = std::mem::size_of_val(data);
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "{} buffer needs {required_size} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}",
                self.label
            );
            let new_buffer_size = data
                .len()
                .next_power_of_two()
                .min(MAX_INSTANCE_BUFFER_SIZE / element_size);
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer(device_context, &self.buffer, data)
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            topology,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        sampler: &[Option<ID3D11SamplerState>],
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));

            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range(
        &self,
        device_context: &ID3D11DeviceContext,
        batch_params_buffer: &ID3D11Buffer,
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        batch_params_buffer: &ID3D11Buffer,
        sampler: &[Option<ID3D11SamplerState>],
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct PathRasterizationSprite {
    pub(crate) xy_position: Point<ScaledPixels>,
    pub(crate) st_position: Point<f32>,
    pub(crate) color: Background,
    pub(crate) bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct PathSprite {
    pub(crate) bounds: Bounds<ScaledPixels>,
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

pub(crate) fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    ID3D11Texture2D,
    Option<ID3D11ShaderResourceView>,
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device)?;
    let (path_intermediate_texture, path_intermediate_srv) =
        create_path_intermediate_texture(&devices.device, width, height)?;
    let (path_intermediate_msaa_texture, path_intermediate_msaa_view) =
        create_path_intermediate_msaa_texture_and_view(&devices.device, width, height)?;
    let viewport = D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    Ok((
        render_target,
        render_target_view,
        path_intermediate_texture,
        path_intermediate_srv,
        path_intermediate_msaa_texture,
        path_intermediate_msaa_view,
        viewport,
    ))
}

#[inline]
pub(crate) fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = unsafe { swap_chain.GetBuffer(0) }?;
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
pub(crate) fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11ShaderResourceView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };

    Ok((texture, Some(shader_resource_view.unwrap())))
}

/// Create a color texture usable as both a render target and a shader resource, returning both
/// views. Used for the blur offscreen targets.
#[inline]
fn create_color_target(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    Option<ID3D11ShaderResourceView>,
)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width.max(1),
            Height: height.max(1),
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut rtv = None;
    unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut rtv))? };
    let mut srv = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut srv))? };
    Ok((texture, rtv, srv))
}

#[inline]
pub(crate) fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

#[inline]
fn set_rasterizer_state(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Result<()> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: false.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let rasterizer_state = unsafe {
        let mut state = None;
        device.CreateRasterizerState(&desc, Some(&mut state))?;
        state.unwrap()
    };
    unsafe { device_context.RSSetState(&rasterizer_state) };
    Ok(())
}

// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ns-d3d11-d3d11_blend_desc
#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_rasterization(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_sprite(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    // If the feature level is set to greater than D3D_FEATURE_LEVEL_9_3, the display
    // device performs the blend in linear space, which is ideal.
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

/// Create a CPU-writable dynamic constant buffer of the given byte size (rounded up to 16).
#[inline]
/// Creates a dynamic constant buffer sized at runtime.
///
/// Distinct from the generic [`create_constant_buffer`], which derives its size from a `T` known
/// at compile time — the blur passes size their cbuffer from a plain byte count.
fn create_constant_buffer_sized(device: &ID3D11Device, byte_size: usize) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: byte_size.next_multiple_of(16) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        ..Default::default()
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

/// A blend state that overwrites the target (no blending) — used for the blur downsample and
/// gaussian passes.
#[inline]
fn create_blend_state_no_blend(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = false.into();
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_vertex_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_fragment_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_constant_buffer<T>(device: &ID3D11Device) -> Result<Option<ID3D11Buffer>> {
    const { assert!(std::mem::size_of::<T>() != 0 && std::mem::size_of::<T>().is_multiple_of(16)) };
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: std::mem::size_of::<T>() as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer)
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: (element_size * buffer_size) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
        StructureByteStride: element_size as u32,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, None, Some(&mut view)) }?;
    Ok(view)
}

#[inline]
pub(crate) fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

#[inline]
pub(crate) fn update_batch_start(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    first_instance: u32,
) -> Result<()> {
    update_buffer(
        device_context,
        buffer,
        &[BatchParams {
            start_index: first_instance,
            _padding: [0; 3],
        }],
    )
}

#[inline]
fn set_pipeline_state(
    device_context: &ID3D11DeviceContext,
    buffer_view: &[Option<ID3D11ShaderResourceView>],
    topology: D3D_PRIMITIVE_TOPOLOGY,
    vertex_shader: &ID3D11VertexShader,
    fragment_shader: &ID3D11PixelShader,
    blend_state: &ID3D11BlendState,
) {
    unsafe {
        device_context.VSSetShaderResources(1, Some(buffer_view));
        device_context.PSSetShaderResources(1, Some(buffer_view));
        device_context.IASetPrimitiveTopology(topology);
        device_context.VSSetShader(vertex_shader, None);
        device_context.PSSetShader(fragment_shader, None);
        device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
    }
}

/// Select the colour space DXGI presents the swap chain in. Composition swap chains support this
/// too (checked on Windows 11 with an FP16 one), so the support check is real, not a formality.
fn set_swap_chain_color_space(
    swap_chain: &IDXGISwapChain1,
    color_space: DXGI_COLOR_SPACE_TYPE,
) -> Result<()> {
    let swap_chain: IDXGISwapChain3 = swap_chain.cast()?;
    let support = unsafe { swap_chain.CheckColorSpaceSupport(color_space) }?;
    anyhow::ensure!(
        support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32 != 0,
        "swap chain cannot present in colour space {color_space:?}"
    );
    unsafe { swap_chain.SetColorSpace1(color_space) }?;
    Ok(())
}

/// A CPU copy of a texture's first mip, rows packed tight.
struct TextureReadback {
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    bytes: Vec<u8>,
}

impl TextureReadback {
    /// Every channel as `f32`, in the texture's channel order (BGRA stays BGRA): 8-bit channels
    /// over 255, half floats as stored.
    fn rgba_f32(&self) -> Vec<f32> {
        if self.format == HDR_RENDER_TARGET_FORMAT {
            self.bytes
                .chunks_exact(2)
                .map(|half| half_to_f32(u16::from_le_bytes([half[0], half[1]])))
                .collect()
        } else {
            self.bytes.iter().map(|&v| v as f32 / 255.0).collect()
        }
    }
}

/// Copy `texture` through a `D3D11_USAGE_STAGING` texture into CPU memory.
fn read_back_texture(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    texture: &ID3D11Texture2D,
) -> Result<TextureReadback> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { texture.GetDesc(&mut desc) };
    let width = desc.Width;
    let height = desc.Height;
    let bytes_per_pixel = if desc.Format == HDR_RENDER_TARGET_FORMAT {
        8
    } else {
        4
    };
    let staging_desc = D3D11_TEXTURE2D_DESC {
        Usage: D3D11_USAGE_STAGING,
        BindFlags: 0,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
        MiscFlags: 0,
        MipLevels: 1,
        ArraySize: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..desc
    };
    let mut staging: Option<ID3D11Texture2D> = None;
    unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging))? };
    let staging = staging.context("creating staging texture")?;
    unsafe { context.CopyResource(&staging, texture) };

    let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
    unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };
    let row_bytes = width as usize * bytes_per_pixel;
    let mut bytes = vec![0u8; row_bytes * height as usize];
    // SAFETY: `Map` succeeded, so `pData` points at `RowPitch * height`
    // readable bytes for as long as the mapping is held, and `RowPitch >=
    // row_bytes` (it only ever adds trailing padding). `bytes` is sized
    // `row_bytes * height`, so every copy stays in bounds on both sides,
    // and the regions cannot overlap (`bytes` is a fresh allocation).
    unsafe {
        let src = mapped.pData as *const u8;
        for row in 0..height as usize {
            let s = src.add(row * mapped.RowPitch as usize);
            let d = bytes.as_mut_ptr().add(row * row_bytes);
            std::ptr::copy_nonoverlapping(s, d, row_bytes);
        }
        context.Unmap(&staging, 0);
    }
    Ok(TextureReadback {
        width,
        height,
        format: desc.Format,
        bytes,
    })
}

/// One premultiplied scRGB pixel (RGBA, SDR white at `scale`) back to the premultiplied sRGB
/// bytes the 8-bit pipeline would have drawn — the inverse of `hdr_output_fragment`, with HDR
/// highlights clipped to white.
#[cfg(feature = "test-support")]
fn scrgb_to_srgb8(px: &[f32], scale: f32) -> [u8; 4] {
    let alpha = px[3].clamp(0.0, 1.0);
    let encode = |linear_premultiplied: f32| {
        if alpha <= 0.0 {
            return 0;
        }
        let linear = (linear_premultiplied / (scale * alpha)).clamp(0.0, 1.0);
        let encoded = if linear <= 0.0031308 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (encoded * alpha * 255.0).round() as u8
    };
    [
        encode(px[0]),
        encode(px[1]),
        encode(px[2]),
        (alpha * 255.0).round() as u8,
    ]
}

/// IEEE 754 binary16 → f32 (no `half` dependency; readback only).
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = match exp {
        0 => (frac as f32) * 2f32.powi(-24),
        0x1f => {
            if frac == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -f } else { f }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

pub(crate) const BUFFER_COUNT: usize = 3;

pub(crate) mod shader_resources {
    use anyhow::Result;

    #[cfg(debug_assertions)]
    use windows::{
        Win32::Graphics::Direct3D::{
            Fxc::{D3DCOMPILE_DEBUG, D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompileFromFile},
            ID3DBlob,
        },
        core::{HSTRING, PCSTR},
    };

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        Shadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        BlurDownsample,
        Blur,
        BlurComposite,
        HdrOutput,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderTarget {
        Vertex,
        Fragment,
    }

    pub(crate) struct RawShaderBytes<'t> {
        inner: &'t [u8],

        #[cfg(debug_assertions)]
        _blob: ID3DBlob,
    }

    impl<'t> RawShaderBytes<'t> {
        pub(crate) fn new(module: ShaderModule, target: ShaderTarget) -> Result<Self> {
            #[cfg(not(debug_assertions))]
            {
                Ok(Self::from_bytes(module, target))
            }
            #[cfg(debug_assertions)]
            {
                let blob = build_shader_blob(module, target)?;
                let inner = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    )
                };
                Ok(Self { inner, _blob: blob })
            }
        }

        pub(crate) fn as_bytes(&'t self) -> &'t [u8] {
            self.inner
        }

        #[cfg(not(debug_assertions))]
        fn from_bytes(module: ShaderModule, target: ShaderTarget) -> Self {
            let bytes = match module {
                ShaderModule::Quad => match target {
                    ShaderTarget::Vertex => QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::Shadow => match target {
                    ShaderTarget::Vertex => SHADOW_VERTEX_BYTES,
                    ShaderTarget::Fragment => SHADOW_FRAGMENT_BYTES,
                },
                ShaderModule::Underline => match target {
                    ShaderTarget::Vertex => UNDERLINE_VERTEX_BYTES,
                    ShaderTarget::Fragment => UNDERLINE_FRAGMENT_BYTES,
                },
                ShaderModule::PathRasterization => match target {
                    ShaderTarget::Vertex => PATH_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::PathSprite => match target {
                    ShaderTarget::Vertex => PATH_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::MonochromeSprite => match target {
                    ShaderTarget::Vertex => MONOCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MONOCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::SubpixelSprite => match target {
                    ShaderTarget::Vertex => SUBPIXEL_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SUBPIXEL_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::PolychromeSprite => match target {
                    ShaderTarget::Vertex => POLYCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => POLYCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::BlurDownsample => match target {
                    ShaderTarget::Vertex => BLUR_DOWNSAMPLE_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_DOWNSAMPLE_FRAGMENT_BYTES,
                },
                ShaderModule::Blur => match target {
                    ShaderTarget::Vertex => BLUR_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_FRAGMENT_BYTES,
                },
                ShaderModule::BlurComposite => match target {
                    ShaderTarget::Vertex => BLUR_COMPOSITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => BLUR_COMPOSITE_FRAGMENT_BYTES,
                },
                ShaderModule::HdrOutput => match target {
                    ShaderTarget::Vertex => HDR_OUTPUT_VERTEX_BYTES,
                    ShaderTarget::Fragment => HDR_OUTPUT_FRAGMENT_BYTES,
                },
            };
            Self { inner: bytes }
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn build_shader_blob(entry: ShaderModule, target: ShaderTarget) -> Result<ID3DBlob> {
        unsafe {
            use windows::Win32::Graphics::{
                Direct3D::ID3DInclude, Hlsl::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            };

            let entry = format!(
                "{}_{}\0",
                entry.as_str(),
                match target {
                    ShaderTarget::Vertex => "vertex",
                    ShaderTarget::Fragment => "fragment",
                }
            );
            let target = match target {
                ShaderTarget::Vertex => "vs_4_1\0",
                ShaderTarget::Fragment => "ps_4_1\0",
            };

            let mut compile_blob = None;
            let mut error_blob = None;
            let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("src/shaders.hlsl")
                .canonicalize()?;
            let shader_path = crate::fast::rounded_clip::shader_path(shader_path, env!("OUT_DIR"));

            let entry_point = PCSTR::from_raw(entry.as_ptr());
            let target_cstr = PCSTR::from_raw(target.as_ptr());

            // really dirty trick because winapi bindings are unhappy otherwise
            let include_handler = &std::mem::transmute::<usize, ID3DInclude>(
                D3D_COMPILE_STANDARD_FILE_INCLUDE as usize,
            );

            let ret = D3DCompileFromFile(
                &HSTRING::from(shader_path.to_str().unwrap()),
                None,
                include_handler,
                entry_point,
                target_cstr,
                D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
                0,
                &mut compile_blob,
                Some(&mut error_blob),
            );
            if ret.is_err() {
                let Some(error_blob) = error_blob else {
                    return Err(anyhow::anyhow!("{ret:?}"));
                };

                let error_string =
                    std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8)
                        .to_string_lossy();
                log::error!("Shader compile error: {}", error_string);
                return Err(anyhow::anyhow!("Compile error: {}", error_string));
            }
            Ok(compile_blob.unwrap())
        }
    }

    #[cfg(not(debug_assertions))]
    include!(concat!(env!("OUT_DIR"), "/shaders_bytes.rs"));

    #[cfg(debug_assertions)]
    impl ShaderModule {
        pub fn as_str(self) -> &'static str {
            match self {
                ShaderModule::Quad => "quad",
                ShaderModule::Shadow => "shadow",
                ShaderModule::Underline => "underline",
                ShaderModule::PathRasterization => "path_rasterization",
                ShaderModule::PathSprite => "path_sprite",
                ShaderModule::MonochromeSprite => "monochrome_sprite",
                ShaderModule::SubpixelSprite => "subpixel_sprite",
                ShaderModule::PolychromeSprite => "polychrome_sprite",
                ShaderModule::BlurDownsample => "blur_downsample",
                ShaderModule::Blur => "blur",
                ShaderModule::BlurComposite => "blur_composite",
                ShaderModule::HdrOutput => "hdr_output",
            }
        }
    }
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}

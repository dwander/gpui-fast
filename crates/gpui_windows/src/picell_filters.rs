//! [PiCell] Blur filters (CSS `backdrop-filter` / `filter`, `Styled::appear_scale`) and the HDR
//! output pass, drawn from either batch loop: gpui-fast's (`fast::frame::draw_scene_masked`),
//! which draws every frame, or upstream's labeled one in `DirectXRenderer::render`, which runs
//! while a graphics debugger captures.
//!
//! Kept out of `directx_renderer.rs` (and out of gpui-fast's `fast/`) so both loops share one
//! implementation and upstream merges touch as little of it as possible.
//!
//! While a frame holds a filter, or HDR output is on, the scene is drawn into the offscreen
//! `scene_color` target rather than the swap chain, so a filter can sample what is already
//! painted under it; [`FilterTargets::finish`] then copies (or, for HDR, converts) it into the
//! swap chain.

use std::slice;

use anyhow::{Context as _, Result};
use gpui::{
    BackdropFilter, Bounds, Corners, FilterBoundary, Point, ScaledFilter, ScaledPixels, Scene,
    Size,
};
use windows::Win32::Graphics::{
    Direct3D::{
        D3D_PRIMITIVE_TOPOLOGY, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
    },
    Direct3D11::{
        D3D11_VIEWPORT, ID3D11BlendState,
        ID3D11Buffer, ID3D11DeviceContext, ID3D11PixelShader, ID3D11RenderTargetView,
        ID3D11SamplerState, ID3D11ShaderResourceView, ID3D11VertexShader,
    },
};

use crate::DirectXRenderer;

/// Mirrors the `BlurParams` cbuffer (register b3) in `shaders.hlsl`. 96 bytes (a multiple of 16,
/// as constant buffers require). Updated per blur pass.
#[repr(C)]
#[derive(Clone, Copy)]
struct BlurParams {
    bounds: Bounds<ScaledPixels>,
    content_mask: Bounds<ScaledPixels>,
    corner_radii: [f32; 4],
    direction: [f32; 2],
    sigma: f32,
    opacity: f32,
    tap_count: f32,
    /// 1.0 clips the composite to the rounded rect (backdrop); 0.0 lets content blur bleed past
    /// its bounds like CSS `filter: blur`.
    clip_rounded: f32,
    /// 1.0 = snapped 2:1 box downsample (anchor the half-res grid to a fixed 2px grid at the
    /// origin, so a stationary element blurs identically at every window size); 0.0 = 1:1 copy
    /// (the scene blit, which must not downsample). Downsample pass only.
    downsample: f32,
    /// Spacing between taps in pixels (gaussian passes only); >1 lets `tap_count` taps span very
    /// large radii without truncating the gaussian, matching the wgpu backend.
    tap_step: f32,
    /// Composite pass only: resample the source about `scale_anchor` by this factor (1.0 = 1:1).
    /// Drives `Styled::appear_scale`.
    scale: f32,
    /// Composite pass only: how many source texels one screen pixel spans — 2.0 for the half-res
    /// blur textures, 1.0 when compositing a full-resolution (unblurred) group.
    source_texel_scale: f32,
    /// Composite pass only: the point `scale` scales about, in scene pixels. Rounds the struct
    /// out to a multiple of 16 bytes, as constant buffers require.
    scale_anchor: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<BlurParams>() == 96);

/// The size of the blur constant buffer the renderer creates.
pub(crate) const BLUR_PARAMS_SIZE: usize = std::mem::size_of::<BlurParams>();

impl Default for BlurParams {
    fn default() -> Self {
        BlurParams {
            bounds: Bounds::default(),
            content_mask: Bounds::default(),
            corner_radii: [0.0; 4],
            direction: [0.0, 0.0],
            sigma: 0.0,
            opacity: 1.0,
            tap_count: 0.0,
            clip_rounded: 0.0,
            downsample: 0.0,
            tap_step: 0.0,
            scale: 1.0,
            source_texel_scale: 2.0,
            scale_anchor: [0.0; 2],
        }
    }
}

/// The largest blur radius in a scene-space filter chain, in device pixels — used to size the
/// blur kernel and the dilated region the blur passes are scissored to.
///
/// The `match` is exhaustive on purpose: adding a [`ScaledFilter`] variant breaks it here,
/// forcing this backend to handle (or deliberately ignore) the new filter rather than silently
/// dropping it.
fn max_blur_radius(filters: &[ScaledFilter]) -> f32 {
    filters.iter().fold(0.0, |acc, filter| match filter {
        ScaledFilter::Blur(radius) => acc.max(radius.0),
    })
}

/// Grow (or shrink) `bounds` about `anchor` by `factor`.
fn scale_about(
    bounds: Bounds<ScaledPixels>,
    factor: f32,
    anchor: Point<ScaledPixels>,
) -> Bounds<ScaledPixels> {
    Bounds {
        origin: Point {
            x: ScaledPixels(anchor.x.0 + (bounds.origin.x.0 - anchor.x.0) * factor),
            y: ScaledPixels(anchor.y.0 + (bounds.origin.y.0 - anchor.y.0) * factor),
        },
        size: Size {
            width: ScaledPixels(bounds.size.width.0 * factor),
            height: ScaledPixels(bounds.size.height.0 * factor),
        },
    }
}

/// Flatten a `Corners` into the `[tl, tr, br, bl]` order expected by the blur composite shader.
fn corner_radii_array(corners: Corners<ScaledPixels>) -> [f32; 4] {
    [
        corners.top_left.0,
        corners.top_right.0,
        corners.bottom_right.0,
        corners.bottom_left.0,
    ]
}

/// Whether `scene` holds a filter anywhere, its rounded clip groups included: gpui-fast moves a
/// rounded clip's content into a scene of its own (`fast::rounded_clip_scene::compile`), so the
/// top-level lists miss the filters drawn inside one.
pub(crate) fn scene_has_filters(scene: &Scene) -> bool {
    !scene.backdrop_filters.is_empty()
        || !scene.filter_boundaries.is_empty()
        || scene
            .layers
            .frames
            .iter()
            .filter(|frame| frame.tile_size == 0)
            .filter_map(|frame| frame.content.scene())
            .any(scene_has_filters)
}

/// What the blur passes draw with: the renderer's blur pipelines and scratch targets, cloned
/// (each clone an AddRef) so a pass can run while the batch loop holds the renderer's other
/// fields.
pub(crate) struct BlurPass {
    context: ID3D11DeviceContext,
    downsample_vertex: ID3D11VertexShader,
    downsample_fragment: ID3D11PixelShader,
    blur_vertex: ID3D11VertexShader,
    blur_fragment: ID3D11PixelShader,
    composite_vertex: ID3D11VertexShader,
    composite_fragment: ID3D11PixelShader,
    hdr_output_vertex: ID3D11VertexShader,
    hdr_output_fragment: ID3D11PixelShader,
    params_buffer: ID3D11Buffer,
    blend_replace: ID3D11BlendState,
    blend_composite: ID3D11BlendState,
    global_params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
    ping_rtv: Option<ID3D11RenderTargetView>,
    ping_srv: Option<ID3D11ShaderResourceView>,
    pong_rtv: Option<ID3D11RenderTargetView>,
    pong_srv: Option<ID3D11ShaderResourceView>,
    full_viewport: D3D11_VIEWPORT,
    width: u32,
    height: u32,
}

impl BlurPass {
    pub(crate) fn new(renderer: &DirectXRenderer) -> Result<Self> {
        let devices = renderer.devices.as_ref().context("devices missing")?;
        let resources = renderer.resources.as_ref().context("resources missing")?;
        let pipelines = &renderer.pipelines;
        let blur = &resources.blur;
        Ok(Self {
            context: devices.device_context.clone(),
            downsample_vertex: pipelines.blur_downsample_vertex.clone(),
            downsample_fragment: pipelines.blur_downsample_fragment.clone(),
            blur_vertex: pipelines.blur_vertex.clone(),
            blur_fragment: pipelines.blur_fragment.clone(),
            composite_vertex: pipelines.blur_composite_vertex.clone(),
            composite_fragment: pipelines.blur_composite_fragment.clone(),
            hdr_output_vertex: pipelines.hdr_output_vertex.clone(),
            hdr_output_fragment: pipelines.hdr_output_fragment.clone(),
            params_buffer: pipelines.blur_params_buffer.clone(),
            blend_replace: pipelines.blur_blend_replace.clone(),
            blend_composite: pipelines.blur_blend_composite.clone(),
            global_params_buffer: renderer.globals.global_params_buffer.clone(),
            sampler: renderer.globals.sampler.clone(),
            ping_rtv: blur.ping_rtv.clone(),
            ping_srv: blur.ping_srv.clone(),
            pong_rtv: blur.pong_rtv.clone(),
            pong_srv: blur.pong_srv.clone(),
            full_viewport: resources.viewport,
            width: renderer.width,
            height: renderer.height,
        })
    }

    /// Run a single blur pass: a full-screen (or composite) draw sampling `source_srv` into
    /// `target_rtv`, with `params` in the blur constant buffer (b3).
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        vertex: &ID3D11VertexShader,
        fragment: &ID3D11PixelShader,
        blend: &ID3D11BlendState,
        target_rtv: &Option<ID3D11RenderTargetView>,
        source_srv: &Option<ID3D11ShaderResourceView>,
        params: BlurParams,
        viewport: &D3D11_VIEWPORT,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        clear: bool,
    ) -> Result<()> {
        let ctx = &self.context;
        crate::directx_renderer::update_buffer(ctx, &self.params_buffer, &[params])?;
        let null_srv: [Option<ID3D11ShaderResourceView>; 1] = [None];
        let blur_params = [Some(self.params_buffer.clone())];
        unsafe {
            // Unbind any SRV at slot 0 so the target texture isn't simultaneously bound as input.
            ctx.PSSetShaderResources(0, Some(&null_srv));
            if clear {
                ctx.ClearRenderTargetView(
                    target_rtv.as_ref().context("blur target view missing")?,
                    &[0.0; 4],
                );
            }
            ctx.OMSetRenderTargets(Some(slice::from_ref(target_rtv)), None);
            ctx.RSSetViewports(Some(slice::from_ref(viewport)));
            ctx.IASetPrimitiveTopology(topology);
            ctx.VSSetShader(vertex, None);
            ctx.PSSetShader(fragment, None);
            ctx.VSSetConstantBuffers(0, Some(slice::from_ref(&self.global_params_buffer)));
            ctx.PSSetConstantBuffers(0, Some(slice::from_ref(&self.global_params_buffer)));
            // Slot 3, not 1 — slot 1 holds BatchParams for the instanced draws, and slot 2
            // gpui-fast's rounded clip parameters (see shaders.hlsl).
            ctx.VSSetConstantBuffers(3, Some(&blur_params));
            ctx.PSSetConstantBuffers(3, Some(&blur_params));
            ctx.PSSetSamplers(0, Some(slice::from_ref(&self.sampler)));
            ctx.PSSetShaderResources(0, Some(slice::from_ref(source_srv)));
            ctx.OMSetBlendState(blend, None, 0xFFFFFFFF);
            ctx.DrawInstanced(vertex_count, 1, 0, 0);
            // Unbind the source so the target can be rebound as a render target next.
            ctx.PSSetShaderResources(0, Some(&null_srv));
        }
        Ok(())
    }

    /// Blur `source_srv` (full-resolution) using the half-res ping/pong textures and composite the
    /// result into `target_rtv`, clipped to `bounds`/`corner_radii`/`content_mask`, resampled by
    /// `scale` about `scale_anchor`, and modulated by `opacity`. Shared by the backdrop and
    /// content-filter paths.
    #[allow(clippy::too_many_arguments)]
    fn blur_and_composite(
        &self,
        source_srv: &Option<ID3D11ShaderResourceView>,
        target_rtv: &Option<ID3D11RenderTargetView>,
        bounds: Bounds<ScaledPixels>,
        content_mask: Bounds<ScaledPixels>,
        corner_radii: [f32; 4],
        blur_radius: f32,
        opacity: f32,
        scale: f32,
        scale_anchor: Point<ScaledPixels>,
        // Backdrop clips to the rounded rect; content (`filter`) bleeds past its bounds.
        clip_rounded: bool,
    ) -> Result<()> {
        // Sigma is halved because the blur runs at half resolution.
        let sigma = (blur_radius * 0.5).max(0.0);
        // A group can be isolated to be scaled rather than blurred; then there is nothing to blur
        // and the composite samples the full-resolution source directly (a half-res round trip
        // would soften it for no reason).
        let blurred = sigma > 0.0;
        if !blurred && scale == 1.0 {
            return Ok(());
        }
        // Span ±3σ. If that needs more than 32 taps, spread the taps apart (tap_step > 1) rather
        // than truncating the kernel — keeps very large radii from clipping. Matches wgpu.
        let ideal_taps = (3.0 * sigma).ceil();
        let tap_count = ideal_taps.clamp(1.0, 32.0);
        let tap_step = (ideal_taps / tap_count).max(1.0);
        // Content blur bleeds ~3·radius past the box, so its composite quad covers a dilated rect;
        // a group scaled up past 1 likewise reaches past its bounds. Both grow the quad only —
        // the shape still comes from the sampled alpha.
        //
        // An unblurred (scale-only) group composites wherever its source can land inside the
        // clip instead. Its children may paint outside its box — an outset ring, a badge — and
        // the group target holds exactly what the group painted (it is cleared to transparent),
        // so clipping to the box would cut that overflow off for as long as the group is
        // isolated, and it would pop in only when isolation ends (an `appear_scale` entrance
        // whose frosted rim appeared late). The quad stops where the source leaves the content
        // mask: past it the target is empty anyway, and the sampler wraps, so reading past the
        // target's edge would pull in the opposite side. A blurred group keeps the dilated box —
        // it stays isolated for as long as it exists, and a full-clip pass every frame would
        // cost more than the overflow it recovers.
        let composite_bounds = if clip_rounded {
            bounds
        } else if !blurred {
            content_mask.intersect(&scale_about(content_mask, scale, scale_anchor))
        } else {
            let bleed = bounds.dilate(ScaledPixels(3.0 * blur_radius));
            if scale > 1.0 {
                scale_about(bleed, scale, scale_anchor)
            } else {
                bleed
            }
        };
        let half_w = (self.width / 2).max(1);
        let half_h = (self.height / 2).max(1);
        let half_vp = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: half_w as f32,
            Height: half_h as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };

        // Downsample source -> ping, then separable gaussian ping -> pong -> ping.
        if blurred {
            self.run(
                &self.downsample_vertex,
                &self.downsample_fragment,
                &self.blend_replace,
                &self.ping_rtv,
                source_srv,
                BlurParams {
                    downsample: 1.0,
                    ..Default::default()
                },
                &half_vp,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                3,
                true,
            )?;
            self.run(
                &self.blur_vertex,
                &self.blur_fragment,
                &self.blend_replace,
                &self.pong_rtv,
                &self.ping_srv,
                BlurParams {
                    direction: [1.0 / half_w as f32, 0.0],
                    sigma,
                    tap_count,
                    tap_step,
                    ..Default::default()
                },
                &half_vp,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                3,
                true,
            )?;
            self.run(
                &self.blur_vertex,
                &self.blur_fragment,
                &self.blend_replace,
                &self.ping_rtv,
                &self.pong_srv,
                BlurParams {
                    direction: [0.0, 1.0 / half_h as f32],
                    sigma,
                    tap_count,
                    tap_step,
                    ..Default::default()
                },
                &half_vp,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                3,
                true,
            )?;
        }
        // Composite the (blurred) result into the target, preserving its contents. The blurred
        // result lives in the half-res ping texture; an unblurred group is sampled at full res.
        self.run(
            &self.composite_vertex,
            &self.composite_fragment,
            &self.blend_composite,
            target_rtv,
            if blurred { &self.ping_srv } else { source_srv },
            BlurParams {
                bounds: composite_bounds,
                content_mask,
                corner_radii,
                opacity,
                clip_rounded: if clip_rounded { 1.0 } else { 0.0 },
                scale,
                source_texel_scale: if blurred { 2.0 } else { 1.0 },
                scale_anchor: [scale_anchor.x.0, scale_anchor.y.0],
                ..Default::default()
            },
            &self.full_viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            4,
            false,
        )
    }

    /// Copy the offscreen scene texture into the swapchain render target, converting it to scRGB
    /// when `hdr_output` (see `HdrOutput` in `directx_renderer.rs`).
    fn blit(
        &self,
        source_srv: &Option<ID3D11ShaderResourceView>,
        target_rtv: &Option<ID3D11RenderTargetView>,
        hdr_output: bool,
    ) -> Result<()> {
        let (vertex, fragment) = if hdr_output {
            (&self.hdr_output_vertex, &self.hdr_output_fragment)
        } else {
            (&self.downsample_vertex, &self.downsample_fragment)
        };
        self.run(
            vertex,
            fragment,
            &self.blend_replace,
            target_rtv,
            source_srv,
            BlurParams::default(),
            &self.full_viewport,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            3,
            true,
        )
    }
}

/// The render targets a frame with filters is drawn through: the offscreen scene, one isolation
/// target per content-filter nesting level, and the stack of what each open group draws over.
pub(crate) struct FilterTargets {
    pass: BlurPass,
    /// Where the batches draw right now — the scene, or the innermost open group. Path batches
    /// restore to it after their own pass.
    pub(crate) current_rtv: Option<ID3D11RenderTargetView>,
    current_srv: Option<ID3D11ShaderResourceView>,
    group_rtvs: Vec<Option<ID3D11RenderTargetView>>,
    group_srvs: Vec<Option<ID3D11ShaderResourceView>>,
    /// (parent_rtv, parent_srv, isolated) for every open group.
    stack: Vec<(
        Option<ID3D11RenderTargetView>,
        Option<ID3D11ShaderResourceView>,
        bool,
    )>,
    scene_srv: Option<ID3D11ShaderResourceView>,
    swapchain_rtv: Option<ID3D11RenderTargetView>,
    hdr_output: bool,
}

impl FilterTargets {
    /// Binds the cleared offscreen scene as the frame's target, when the frame needs it: it holds
    /// a filter, or HDR output is on. `swapchain_clear` is what the swap chain was cleared to;
    /// HDR stands in for the swap chain, so it starts from that, while the blur path keeps a
    /// transparent scene.
    pub(crate) fn begin(
        renderer: &DirectXRenderer,
        scene: &Scene,
        hdr_output: bool,
        swapchain_clear: [f32; 4],
    ) -> Result<Option<Self>> {
        if !hdr_output && !scene_has_filters(scene) {
            return Ok(None);
        }
        let pass = BlurPass::new(renderer)?;
        let resources = renderer.resources.as_ref().context("resources missing")?;
        let blur = &resources.blur;
        let scene_rtv = blur.scene_color_rtv.clone();
        let clear = if hdr_output { swapchain_clear } else { [0.0; 4] };
        unsafe {
            if let Some(rtv) = scene_rtv.as_ref() {
                pass.context.ClearRenderTargetView(rtv, &clear);
            }
            pass.context
                .OMSetRenderTargets(Some(slice::from_ref(&scene_rtv)), None);
        }
        Ok(Some(Self {
            current_rtv: scene_rtv,
            current_srv: blur.scene_color_srv.clone(),
            group_rtvs: blur.group_rtvs.clone(),
            group_srvs: blur.group_srvs.clone(),
            stack: Vec::new(),
            scene_srv: blur.scene_color_srv.clone(),
            swapchain_rtv: resources.render_target_view.clone(),
            hdr_output,
            pass,
        }))
    }

    /// Blurs what is drawn under each of `filters` and composites it back, clipped to its
    /// rounded rect.
    pub(crate) fn backdrop_filters(&mut self, filters: &[BackdropFilter]) -> Result<()> {
        let result = filters.iter().try_for_each(|filter| {
            self.pass.blur_and_composite(
                &self.current_srv,
                &self.current_rtv,
                filter.bounds,
                filter.content_mask.bounds,
                corner_radii_array(filter.corner_radii),
                max_blur_radius(&filter.filters),
                filter.opacity,
                1.0,
                Point::default(),
                true,
            )
        });
        // Restore the current target for subsequent batches.
        self.bind_current();
        result
    }

    /// Opens or closes a content-filter group. Each isolated nesting level draws into its own
    /// group target (indexed by isolation depth); beyond the pool size deeper filters render
    /// inline without isolation rather than corrupting an outer group.
    pub(crate) fn boundary(&mut self, boundary: &FilterBoundary) -> Result<()> {
        if boundary.is_start {
            let depth = self.stack.iter().filter(|entry| entry.2).count();
            if depth < self.group_rtvs.len() {
                self.stack
                    .push((self.current_rtv.clone(), self.current_srv.clone(), true));
                self.current_rtv = self.group_rtvs[depth].clone();
                self.current_srv = self.group_srvs[depth].clone();
                unsafe {
                    if let Some(rtv) = self.current_rtv.as_ref() {
                        self.pass.context.ClearRenderTargetView(rtv, &[0.0; 4]);
                    }
                }
                self.bind_current();
            } else {
                self.stack
                    .push((self.current_rtv.clone(), self.current_srv.clone(), false));
            }
            Ok(())
        } else if let Some((parent_rtv, parent_srv, isolated)) = self.stack.pop() {
            let result = if isolated {
                self.pass.blur_and_composite(
                    &self.current_srv,
                    &parent_rtv,
                    boundary.bounds,
                    boundary.content_mask.bounds,
                    corner_radii_array(boundary.corner_radii),
                    max_blur_radius(&boundary.filters),
                    boundary.opacity,
                    boundary.scale,
                    boundary.scale_anchor,
                    false,
                )
            } else {
                Ok(())
            };
            self.current_rtv = parent_rtv;
            self.current_srv = parent_srv;
            self.bind_current();
            result
        } else {
            Ok(())
        }
    }

    fn bind_current(&self) {
        unsafe {
            self.pass
                .context
                .OMSetRenderTargets(Some(slice::from_ref(&self.current_rtv)), None);
        }
    }

    /// Resolves the offscreen scene into the swap chain — converted to scRGB while HDR output is
    /// on, copied as-is otherwise — and leaves the swap chain bound.
    pub(crate) fn finish(self) -> Result<()> {
        self.pass
            .blit(&self.scene_srv, &self.swapchain_rtv, self.hdr_output)
    }
}

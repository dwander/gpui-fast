//! [PiCell] Blur filters (CSS `backdrop-filter` / `filter`, `Styled::appear_scale`) drawn from
//! either batch loop: upstream's in `MetalRenderer::draw_primitives_to_texture`, or gpui-fast's
//! rounded clip loop (`fast::rounded_clip`), which draws every frame holding a rounded
//! `overflow_hidden` container — on macOS nearly every frame that holds a filter at all, since
//! menus, dialogs and the main window's cards are such containers.
//!
//! The blur passes stay in `metal_renderer.rs`, beside the pipelines they draw with. This is the
//! state both loops share: the offscreen scene a frame with a filter is drawn into, so a filter
//! can sample what is already painted under it; one isolation target per content-filter nesting
//! level; and the stack of what each open group draws over. Mirrors
//! `gpui_windows/src/picell_filters.rs`.
//!
//! The passes need a command buffer free of open encoders, so each call takes the loop's open
//! encoder, ends it, and hands back a new one on the current target. A new encoder starts with
//! gpui-fast's rounded clip slots emptied (`new_command_encoder_for_texture`); the rounded clip
//! loop binds its clip stack again before every batch.

use std::mem;

use anyhow::Result;
use gpui::{BackdropFilter, DevicePixels, FilterBoundary, Point, Scene, Size};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLRenderCommandEncoder, MTLTexture,
};

use crate::metal_renderer::{MetalRenderer, max_blur_radius, new_command_encoder_for_texture};

type Texture = ProtocolObject<dyn MTLTexture>;
type Encoder = Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>;

const TRANSPARENT: MTLClearColor = MTLClearColor {
    red: 0.,
    green: 0.,
    blue: 0.,
    alpha: 0.,
};

/// Whether `scene` holds a filter anywhere, its rounded clip groups included: gpui-fast moves a
/// rounded clip's content into a scene of its own (`fast::rounded_clip_scene::compile`), so the
/// top-level lists miss the filters drawn inside one.
fn scene_has_filters(scene: &Scene) -> bool {
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

/// Where a frame's batches draw right now: the current filter target, or `texture` (the
/// drawable) for a frame without filters.
pub(crate) fn target<'a>(filters: &'a Option<FilterTargets>, texture: &'a Texture) -> &'a Texture {
    filters.as_ref().map_or(texture, |filters| &filters.current)
}

/// The render targets a frame with filters is drawn through.
pub(crate) struct FilterTargets {
    scene_color: Retained<Texture>,
    ping: Retained<Texture>,
    pong: Retained<Texture>,
    groups: Vec<Retained<Texture>>,
    /// The scene, or the innermost open group.
    current: Retained<Texture>,
    /// For every open group: the boundary that opened it, the target it composites back into,
    /// and whether it is isolated (beyond the group pool, deeper groups draw inline).
    stack: Vec<(FilterBoundary, Retained<Texture>, bool)>,
}

impl FilterTargets {
    /// The targets for `scene`, when it holds a filter and the renderer has its offscreen
    /// targets; `None` otherwise, and the frame draws straight to the drawable without blurs.
    pub(crate) fn begin(renderer: &MetalRenderer, scene: &Scene) -> Option<Self> {
        if !scene_has_filters(scene) {
            return None;
        }
        let scene_color = renderer.scene_color_texture.clone()?;
        Some(Self {
            current: scene_color.clone(),
            scene_color,
            ping: renderer.blur_ping_texture.clone()?,
            pong: renderer.blur_pong_texture.clone()?,
            groups: renderer.group_textures.clone(),
            stack: Vec::new(),
        })
    }

    /// Blurs what is drawn under each of `filters` and composites it back, clipped to its
    /// rounded rect.
    pub(crate) fn backdrop_filters(
        &self,
        renderer: &MetalRenderer,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: Encoder,
        filters: &[BackdropFilter],
        viewport_size: Size<DevicePixels>,
    ) -> Result<Encoder> {
        encoder.endEncoding();
        for filter in filters {
            renderer.metal_blur_and_composite(
                command_buffer,
                &self.current,
                &self.current,
                &self.ping,
                &self.pong,
                viewport_size,
                filter.bounds,
                filter.content_mask.bounds,
                filter.corner_radii,
                max_blur_radius(&filter.filters),
                filter.opacity,
                1.0,
                Point::default(),
                true,
            )?;
        }
        new_command_encoder_for_texture(command_buffer, &self.current, viewport_size, None)
    }

    /// Opens or closes a content-filter group. Each isolated nesting level draws into its own
    /// group target (indexed by isolation depth); beyond the pool size deeper filters render
    /// inline without isolation rather than corrupting an outer group.
    pub(crate) fn boundary(
        &mut self,
        renderer: &MetalRenderer,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        encoder: Encoder,
        boundary: &FilterBoundary,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Encoder> {
        if boundary.is_start {
            let depth = self.stack.iter().filter(|entry| entry.2).count();
            let Some(group) = self.groups.get(depth).cloned() else {
                self.stack
                    .push((boundary.clone(), self.current.clone(), false));
                return Ok(encoder);
            };
            encoder.endEncoding();
            let parent = mem::replace(&mut self.current, group);
            self.stack.push((boundary.clone(), parent, true));
            return new_command_encoder_for_texture(
                command_buffer,
                &self.current,
                viewport_size,
                Some(TRANSPARENT),
            );
        }
        // Composites with the opening boundary's geometry, as upstream's loop always has.
        let Some((boundary, parent, true)) = self.stack.pop() else {
            return Ok(encoder);
        };
        encoder.endEncoding();
        let group = mem::replace(&mut self.current, parent);
        renderer.metal_blur_and_composite(
            command_buffer,
            &group,
            &self.current,
            &self.ping,
            &self.pong,
            viewport_size,
            boundary.bounds,
            boundary.content_mask.bounds,
            boundary.corner_radii,
            max_blur_radius(&boundary.filters),
            boundary.opacity,
            boundary.scale,
            boundary.scale_anchor,
            false,
        )?;
        new_command_encoder_for_texture(command_buffer, &self.current, viewport_size, None)
    }

    /// Presents the offscreen scene by copying it into `texture` (the drawable). The loop's
    /// encoder must be ended first.
    pub(crate) fn finish(
        &self,
        renderer: &MetalRenderer,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        texture: &Texture,
        viewport_size: Size<DevicePixels>,
    ) -> Result<()> {
        renderer.copy_scene_texture(command_buffer, texture, &self.scene_color, viewport_size)
    }
}

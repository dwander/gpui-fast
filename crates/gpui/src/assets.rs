use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use std::{
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
    /// Whether the frame is half-float (HDR or wide gamut) — selects the float atlas texture
    /// kind. Part of the key so a float image and an 8-bit image never share a tile.
    pub hdr: bool,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
    /// A single half-float frame (RGBA, 16-bit float per channel, extended sRGB encoding).
    /// When present it replaces `data` for rendering and is uploaded into a float atlas
    /// texture, so values outside 0..1 survive to the (float) render target — as EDR
    /// highlights ([`Self::new_hdr`]) or as colours outside the sRGB gamut
    /// ([`Self::new_wide_gamut`]).
    float: Option<FloatFrame>,
}

/// Pixels of a half-float [`RenderImage`]: RGBA half floats, 8 bytes per pixel, row-major, no
/// padding.
struct FloatFrame {
    size: Size<DevicePixels>,
    bytes: Vec<u8>,
    /// Values outside 0..1 are colours outside the sRGB gamut, not highlights — the renderer
    /// must not compress them into the EDR headroom.
    wide_gamut: bool,
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
            float: None,
        }
    }

    /// Create a single-frame HDR image from RGBA half-float pixels (8 bytes per pixel,
    /// extended sRGB encoding, premultiplied alpha). `bytes.len()` must equal
    /// `width * height * 8`; anything else is treated as an empty image.
    ///
    /// Values above 1.0 are brighter than SDR white; the renderer soft-clips them into the
    /// display's current EDR headroom.
    pub fn new_hdr(size: Size<DevicePixels>, bytes: Vec<u8>) -> Self {
        Self::new_float(size, bytes, false)
    }

    /// Create a single-frame SDR image whose colours reach outside the sRGB gamut (Display P3,
    /// Adobe RGB …), as RGBA half floats in extended sRGB — same layout as [`Self::new_hdr`].
    ///
    /// Values outside 0..1 are colours sRGB cannot express, not highlights: a saturated P3 red
    /// has a red channel above 1.0 yet is darker than white. The renderer therefore never
    /// compresses this image into the EDR headroom, and [`Self::is_hdr`] is false for it.
    pub fn new_wide_gamut(size: Size<DevicePixels>, bytes: Vec<u8>) -> Self {
        Self::new_float(size, bytes, true)
    }

    fn new_float(size: Size<DevicePixels>, bytes: Vec<u8>, wide_gamut: bool) -> Self {
        // One counter for every float image, far from the 8-bit one — ids must never collide.
        static NEXT_ID: AtomicUsize = AtomicUsize::new(usize::MAX / 2);

        let expected = (size.width.0.max(0) as usize) * (size.height.0.max(0) as usize) * 8;
        let float = (expected > 0 && bytes.len() == expected).then_some(FloatFrame {
            size,
            bytes,
            wide_gamut,
        });
        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: SmallVec::new(),
            float,
        }
    }

    /// Whether this image carries HDR pixels (see [`Self::new_hdr`]).
    pub fn is_hdr(&self) -> bool {
        self.float.as_ref().is_some_and(|f| !f.wide_gamut)
    }

    /// Whether this image carries wide-gamut SDR pixels (see [`Self::new_wide_gamut`]).
    pub fn is_wide_gamut(&self) -> bool {
        self.float.as_ref().is_some_and(|f| f.wide_gamut)
    }

    /// Whether [`Self::as_bytes`] holds RGBA half floats (HDR or wide gamut) rather than
    /// 8-bit BGRA.
    pub fn is_half_float(&self) -> bool {
        self.float.is_some()
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        if let Some(float) = &self.float {
            return (frame_index == 0).then_some(float.bytes.as_slice());
        }
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        if let Some(float) = &self.float {
            return if frame_index == 0 { float.size } else { Size::default() };
        }
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        if self.float.is_some() {
            return 1;
        }
        self.data.len()
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .field("float", &self.float.as_ref().map(|f| (f.size, f.wide_gamut)))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }

    #[test]
    fn wide_gamut_image_is_half_float_but_not_hdr() {
        let px = size(DevicePixels(2), DevicePixels(1));
        let bytes = vec![0u8; 2 * 8];
        let hdr = RenderImage::new_hdr(px, bytes.clone());
        let wide = RenderImage::new_wide_gamut(px, bytes);

        assert!(hdr.is_hdr() && hdr.is_half_float() && !hdr.is_wide_gamut());
        assert!(!wide.is_hdr() && wide.is_half_float() && wide.is_wide_gamut());
        assert_eq!(wide.size(0), px);
        assert_eq!(wide.frame_count(), 1);
        // Both kinds draw from one id counter — a shared id would alias their atlas tiles.
        assert_ne!(hdr.id, wide.id);
    }
}

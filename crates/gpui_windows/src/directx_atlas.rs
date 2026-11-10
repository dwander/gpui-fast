use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_BOX,
        D3D11_RESOURCE_MISC_GENERATE_MIPS, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Device,
        ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    },
    Dxgi::Common::*,
};

use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Size,
};

/// 이미지 타일을 공유 아틀라스에 패킹하지 않고 **전용 밉맵 텍스처**로 만드는 최소 변 길이(px).
/// 이 크기 이상의 이미지(뷰어 fit/원본 등)는 밉 체인을 생성해, 화면에서 축소해 그릴 때 GPU
/// 트라이리니어 샘플러가 알맞은 밉 레벨을 골라 모아레 없이 실시간으로 그린다. 썸네일
/// (picell CACHE_SIZE=512)·글리프·아이콘은 이 아래라 기존대로 공유 아틀라스에 패킹된다
/// (전용 텍스처 폭증·드로우콜 배치 저하 방지). 아틀라스 기본 텍스처 크기(1024)와 맞물려,
/// 이보다 작은 타일은 항상 패킹 가능하다.
const MIPMAP_MIN_DIMENSION: i32 = 1024;

pub(crate) struct DirectXAtlas(Mutex<AtlasState<DirectXAtlasTextures>>);

struct DirectXAtlasTextures {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    monochrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    polychrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    subpixel_textures: AtlasTextureList<DirectXAtlasTexture>,
    /// HDR 이미지용 RGBA16F 텍스처 (`AtlasTextureKind::PolychromeHdr`). 8비트 아틀라스와 섞이지 않게
    /// 따로 둔다. Windows 렌더타깃은 아직 8비트라 값이 1.0 에서 잘려 SDR 로 보인다 — 업로드·표시는 된다.
    polychrome_hdr_textures: AtlasTextureList<DirectXAtlasTexture>,
}

struct DirectXAtlasTexture {
    id: AtlasTextureId,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    texture: ID3D11Texture2D,
    view: [Option<ID3D11ShaderResourceView>; 1],
    live_atlas_keys: u32,
    /// 전용 밉맵 텍스처인지 — true 면 업로드 후 `GenerateMips` 를 호출하고, 다른 타일을 이 텍스처에
    /// 패킹하지 않는다(밉맵은 아틀라스 타일 경계에서 이웃 픽셀과 번지므로 이미지 1장을 전용으로 담는다).
    mipmapped: bool,
}

impl DirectXAtlas {
    pub(crate) fn new(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Self {
        DirectXAtlas(Mutex::new(AtlasState::new(DirectXAtlasTextures {
            device: device.clone(),
            device_context: device_context.clone(),
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            subpixel_textures: Default::default(),
            polychrome_hdr_textures: Default::default(),
        })))
    }

    /// Returns the view backing `id`, or `None` once every tile in it has been
    /// removed. A scene can still reference such a texture when a cached view
    /// replays a paint from before the image was dropped, so callers must skip
    /// those sprites rather than assume the texture exists.
    pub(crate) fn get_texture_view(
        &self,
        id: AtlasTextureId,
    ) -> Option<[Option<ID3D11ShaderResourceView>; 1]> {
        let lock = self.0.lock();
        let texture = lock.backend.texture(id)?;
        Some(texture.view.clone())
    }

    pub(crate) fn handle_device_lost(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
    ) {
        let mut lock = self.0.lock();
        lock.clear(|textures| {
            textures.device = device.clone();
            textures.device_context = device_context.clone();
            textures.monochrome_textures = AtlasTextureList::default();
            textures.polychrome_textures = AtlasTextureList::default();
            textures.subpixel_textures = AtlasTextureList::default();
        });
    }
}

impl PlatformAtlas for DirectXAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }
}

impl AtlasBackend for DirectXAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> anyhow::Result<AtlasTile> {
        // 큰 이미지는 전용 밉맵 텍스처로 — 화면에서 축소해 그릴 때 GPU 트라이리니어가 밉 레벨을
        // 골라 모아레 없이 실시간 렌더한다. 썸네일/글리프 등 작은 타일은 기존대로 패킹된다.
        // (상류 `AtlasBackend::insert` 는 키를 넘기지 않는다 — 이미지만 Polychrome 으로 오는
        // 큰 타일이므로 kind + 크기로 가른다. 이모지 글리프도 Polychrome 이지만 이 크기가 될 수 없다.)
        let mipmapped = matches!(kind, AtlasTextureKind::Polychrome)
            && size.width.0.max(size.height.0) >= MIPMAP_MIN_DIMENSION;
        let tile = self
            .allocate(size, kind, mipmapped)
            .ok_or_else(|| anyhow::anyhow!("failed to allocate"))?;
        let texture = self
            .texture(tile.texture_id)
            .ok_or_else(|| anyhow::anyhow!("allocated tile refers to a missing texture"))?;
        texture.upload(&self.device_context, tile.bounds, bytes);
        // 밉 0(원본) 업로드 후 나머지 밉 레벨을 GPU 로 채운다(전용 텍스처당 1회, 프레임당 아님).
        if texture.mipmapped {
            texture.generate_mips(&self.device_context);
        }
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::PolychromeHdr => &mut self.polychrome_hdr_textures,
        };

        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl DirectXAtlasTextures {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
        mipmapped: bool,
    ) -> Option<AtlasTile> {
        // 밉맵 이미지는 전용 텍스처가 필요하다(패킹하면 밉이 이웃 타일과 번짐) — 기존 텍스처에 끼워넣지
        // 않고 곧바로 새 텍스처를 만든다. 거꾸로, 일반 타일은 밉맵 텍스처에 패킹하지 않는다(가득 채워
        // 여유 공간이 없기도 하지만 명시적으로 걸러 밉맵 텍스처의 전용성을 보장).
        if !mipmapped {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
                AtlasTextureKind::PolychromeHdr => &mut self.polychrome_hdr_textures,
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .filter(|texture| !texture.mipmapped)
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind, mipmapped)?;
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
        mipmapped: bool,
    ) -> Option<&mut DirectXAtlasTexture> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size for DirectX. See:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-resources-limits
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        // 밉맵 전용 텍스처는 이미지 크기 그대로 만든다(패킹 여유 없이 = 다른 타일이 못 들어오게).
        // 일반 아틀라스는 최소 기본 크기(1024)로 키워 여러 타일을 담는다.
        let size = if mipmapped {
            min_size.min(&MAX_ATLAS_SIZE)
        } else {
            min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE)
        };
        let pixel_format;
        let bind_flag;
        let bytes_per_pixel;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = DXGI_FORMAT_R8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 1;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
            AtlasTextureKind::Subpixel => {
                pixel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
            AtlasTextureKind::PolychromeHdr => {
                pixel_format = DXGI_FORMAT_R16G16B16A16_FLOAT;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 8;
            }
        }
        // 밉맵이면 전체 밉 체인(MipLevels=0) + 렌더타깃 바인드 + GENERATE_MIPS 플래그로 만들고
        // (`GenerateMips` 요구사항), 업로드 후 GPU 로 하위 밉을 채운다. 아니면 단일 밉(기존).
        let (mip_levels, bind_flags, misc_flags) = if mipmapped {
            (
                0u32,
                (bind_flag.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                D3D11_RESOURCE_MISC_GENERATE_MIPS.0 as u32,
            )
        } else {
            (1u32, bind_flag.0 as u32, 0u32)
        };
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: mip_levels,
            ArraySize: 1,
            Format: pixel_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flags,
            CPUAccessFlags: 0,
            MiscFlags: misc_flags,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            // This only returns None if the device is lost, which we will recreate later.
            // So it's ok to return None here.
            self.device
                .CreateTexture2D(&texture_desc, None, Some(&mut texture))
                .ok()?;
        }
        let texture = texture.unwrap();

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            AtlasTextureKind::PolychromeHdr => &mut self.polychrome_hdr_textures,
        };
        let index = texture_list.free_list.pop();
        let view = unsafe {
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
                .ok()?;
            [view]
        };
        let atlas_texture = DirectXAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            bytes_per_pixel,
            allocator: etagere::BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            texture,
            view,
            live_atlas_keys: 0,
            mipmapped,
        };
        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut()
        }
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&DirectXAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => &self.subpixel_textures,
            AtlasTextureKind::PolychromeHdr => &self.polychrome_hdr_textures,
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }
}

impl DirectXAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(
        &self,
        device_context: &ID3D11DeviceContext,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        // `UpdateSubresource` reads `row_pitch * height` bytes from `bytes` based on the
        // `D3D11_BOX` below. If the caller hands us a slice shorter than that, the driver would
        // over-read past the end of the source buffer (potentially by multiple megabytes), so bail
        // out instead. This is a first-insert path rather than a per-frame one, so the check is
        // effectively free.
        let row_bytes = bounds.size.width.to_bytes(self.bytes_per_pixel as u8) as usize;
        let expected = row_bytes * bounds.size.height.0.max(0) as usize;
        if bytes.len() < expected {
            log::error!(
                "DirectXAtlasTexture::upload: source slice is {} bytes but the {}x{} region \
                 requires {} bytes; skipping upload to avoid a driver over-read",
                bytes.len(),
                bounds.size.width.0,
                bounds.size.height.0,
                expected,
            );
            return;
        }
        unsafe {
            device_context.UpdateSubresource(
                &self.texture,
                0,
                Some(&D3D11_BOX {
                    left: bounds.left().0 as u32,
                    top: bounds.top().0 as u32,
                    front: 0,
                    right: bounds.right().0 as u32,
                    bottom: bounds.bottom().0 as u32,
                    back: 1,
                }),
                bytes.as_ptr() as _,
                bounds.size.width.to_bytes(self.bytes_per_pixel as u8),
                0,
            );
        }
    }

    /// 밉 0(원본) 업로드 후 나머지 밉 레벨을 GPU 로 채운다 — 전용 밉맵 텍스처(`mipmapped`)에서만
    /// 호출한다. 텍스처 생성 시 `RENDER_TARGET` 바인드 + `GENERATE_MIPS` 플래그가 있어야 동작한다.
    fn generate_mips(&self, device_context: &ID3D11DeviceContext) {
        if let Some(view) = &self.view[0] {
            unsafe {
                device_context.GenerateMips(view);
            }
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn etagere_point_to_device(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams};
    use std::borrow::Cow;
    use windows::Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_WARP,
            Direct3D11::{D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice},
        },
    };

    fn create_atlas() -> Option<DirectXAtlas> {
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut device_context),
            )
        }
        .ok()?;
        Some(DirectXAtlas::new(&device?, &device_context?))
    }

    fn make_image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(image_id),
            frame_index: 0,
            hdr: false,
        })
    }

    fn insert_tile(atlas: &DirectXAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1);
        let big_key_a = make_image_key(2);
        let big_key_b = make_image_key(3);

        let keeper_tile = insert_tile(&atlas, keeper_key, small);
        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);

        let tile_b = insert_tile(&atlas, big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }
}

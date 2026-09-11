//! `LodSplat`:LoD 樹的一個節點(worker 內部表示)。純資料、不碰 js_sys,獨立成檔是為了
//! 讓 `traverse-bench`(wasm32-wasip1 基準)能 `#[path]` 直接 include 同一份定義。

use glam::Vec3;
use half::f16;

#[derive(Debug, Clone, Default)]
pub(crate) struct LodSplat {
    pub(crate) center: [f16; 3],
    pub(crate) size: f16,
    pub(crate) child_start: u32,
    pub(crate) child_count: u16,
}

impl LodSplat {
    pub(crate) fn new_f16(center: [f16; 3], size: f16, child_start: u32, child_count: u16) -> Self {
        Self { center, size, child_start, child_count }
    }

    /// 只有 `lod_traverse::tests` 用得到(建構測試用的樹);release/wasm build 不含
    /// `#[cfg(test)]`,那裡它是真的死碼,故 `allow` 只在非 test build 生效。
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(center: Vec3, size: f32, child_start: u32, child_count: u16) -> Self {
        let center = center.to_array().map(|x| f16::from_f32(x));
        let size = f16::from_f32(size);
        Self::new_f16(center, size, child_start, child_count)
    }

    pub(crate) fn center(&self) -> glam::Vec3A {
        glam::Vec3A::from_array(self.center.map(|x| x.to_f32()))
    }

    pub(crate) fn size(&self) -> f32 {
        self.size.to_f32()
    }
}


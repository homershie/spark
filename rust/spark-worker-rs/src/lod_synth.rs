//! 合成一棵像 `.rad` 的 LoD 樹,給 `lod_traverse` 的原生基準與「改前 == 改後」集合測試用。
//!
//! worker crate 只在 `cfg(test)` 下編譯它;`rust/traverse-bench` 用 `#[path]` include 同一份跑 wasm。存在的理由(方案 C,2026-09-11):微優化要「每項單獨量」,
//! 但瀏覽器裡量一次 = 使用者走 30 秒;這裡的原生數字不等於 wasm,**相對差距**卻可信,
//! 而且能為假 —— 原生沒差的項目直接不做。集合相等靠 `cut_hash` 釘在測試裡的常數。
//!
//! 形狀:BFS 編號(= build-lod 的 chunk 順序)、分支 4–8、子節點尺寸 × 0.6、中心在父節點
//! 範圍內隨機;chunk-space 索引 → page 用**非恆等**映射(`chunk_to_page[c] = pages-1-c`),
//! 逼出 `expand_until` 裡那行 `(child_page << 16) | (child & 0xffff)`。

use ahash::AHashSet;
use glam::{Vec3, Vec3A};

use crate::lod_traverse::{
    limit_key, expand_until, seed_roots, InstanceParams, TraverseCore, TreeView,
};
use crate::lod_splat::LodSplat;

pub(crate) const CHUNK: usize = 65536;

/// xorshift64*,固定種子、不拉 rand。
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub(crate) fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub(crate) fn range(&mut self, lo: u32, hi_inclusive: u32) -> u32 {
        lo + (self.next_u64() % (hi_inclusive - lo + 1) as u64) as u32
    }
}

/// chunk-space 的樹(`child_start` 是 chunk-space 索引)。
pub(crate) struct SynthTree {
    /// 以 chunk-space 索引排列。
    pub(crate) nodes: Vec<LodSplat>,
    pub(crate) parent: Vec<u32>,
    pub(crate) num_chunks: usize,
}

/// BFS 生到 `max_nodes` 顆為止(最後一層的孩子不生 → 它們是葉)。
pub(crate) fn build_synth(seed: u64, max_nodes: usize, root_size: f32) -> SynthTree {
    let mut rng = Rng::new(seed);
    let mut nodes: Vec<LodSplat> = Vec::with_capacity(max_nodes);
    let mut parent: Vec<u32> = Vec::with_capacity(max_nodes);
    // 暫存 f32 中心/尺寸(LodSplat 存 f16,生孩子時用精確值)
    let mut centers: Vec<(Vec3, f32)> = Vec::with_capacity(max_nodes);

    nodes.push(LodSplat::new(Vec3::ZERO, root_size, 0, 0));
    parent.push(u32::MAX);
    centers.push((Vec3::ZERO, root_size));

    let mut head = 0usize;
    while head < nodes.len() && nodes.len() < max_nodes {
        let (c, s) = centers[head];
        let count = rng.range(4, 8) as usize;
        if nodes.len() + count > max_nodes {
            break;
        }
        let child_start = nodes.len() as u32;
        let cs = s * 0.6;
        for _ in 0..count {
            let off = Vec3::new(rng.unit() - 0.5, rng.unit() - 0.5, rng.unit() - 0.5) * s;
            nodes.push(LodSplat::new(c + off, cs, 0, 0));
            parent.push(head as u32);
            centers.push((c + off, cs));
        }
        nodes[head] = LodSplat::new(c, s, child_start, count as u16);
        head += 1;
    }
    let num_chunks = nodes.len().div_ceil(CHUNK);
    SynthTree { nodes, parent, num_chunks }
}

/// 把 chunk-space 的樹鋪成 paged 版:chunk c → page `pages-1-c`(非恆等),回傳
/// (`splats` 以 paged index 排列, `chunk_to_page`, root_page)。
pub(crate) fn page_out(tree: &SynthTree) -> (Vec<LodSplat>, Vec<u32>, u32) {
    let pages = tree.num_chunks;
    let chunk_to_page: Vec<u32> = (0..pages).map(|c| (pages - 1 - c) as u32).collect();
    let mut splats = vec![LodSplat::default(); pages * CHUNK];
    for (i, node) in tree.nodes.iter().enumerate() {
        let page = chunk_to_page[i / CHUNK] as usize;
        splats[page * CHUNK + (i % CHUNK)] = node.clone();
    }
    let root_page = chunk_to_page[0];
    (splats, chunk_to_page, root_page)
}

pub(crate) fn params(origin: Vec3A, forward: Vec3A) -> InstanceParams {
    // 對齊本專案:lodScale 1.2、Spark 原生 foveation 0.2/0.4、錐 90°/120°
    let cone_dot0 = (0.5f32 * 90.0).to_radians().cos();
    let cone_dot = (0.5f32 * 120.0).to_radians().cos().min(cone_dot0);
    InstanceParams {
        origin,
        forward: forward.normalize(),
        lod_scale: 1.2,
        behind_foveate: 0.2,
        cone_foveate: 0.4,
        cone_dot0,
        cone_dot,
    }
}

/// 桌機 2560×1183、垂直 FOV 60°:`2·tan(30°)/1183`。
pub(crate) const PIXEL_SCALE_LIMIT: f32 = 0.000976;

/// 跑到底,回 (cut 的 paged index 集合, num_splats)。
pub(crate) fn run_atomic(
    splats: &[LodSplat], chunk_to_page: &[u32], root_page: u32, p: InstanceParams, max_splats: usize,
) -> (Vec<u32>, usize) {
    let trees = [TreeView { lod_id: 1, splats, chunk_to_page, params: p }];
    let mut core = TraverseCore::default();
    core.reset(max_splats);
    seed_roots(&mut core, &trees, &[root_page]);
    expand_until(&mut core, &trees, max_splats, limit_key(PIXEL_SCALE_LIMIT), &mut || false);
    let mut cut: Vec<u32> = core.snapshot().into_iter().map(|(_, paged)| paged).collect();
    cut.sort_unstable();
    (cut, core.num_splats)
}

/// 集合指紋(排序後 FNV-1a),釘在測試裡當「改前 == 改後」的判準。
pub(crate) fn cut_hash(sorted: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in sorted {
        h ^= v as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// 合法 cut:每片葉的祖先鏈上恰有一個節點在 cut 裡(以 paged index 對照)。
pub(crate) fn assert_valid_cut(tree: &SynthTree, chunk_to_page: &[u32], cut_paged_sorted: &[u32]) {
    let to_paged = |i: usize| (chunk_to_page[i / CHUNK] << 16) | (i % CHUNK) as u32;
    let set: AHashSet<u32> = cut_paged_sorted.iter().copied().collect();
    assert_eq!(set.len(), cut_paged_sorted.len(), "cut 內有重複");
    for i in 0..tree.nodes.len() {
        if tree.nodes[i].child_count != 0 {
            continue;
        }
        let mut n = i;
        let mut covered = 0;
        loop {
            if set.contains(&to_paged(n)) {
                covered += 1;
            }
            if tree.parent[n] == u32::MAX {
                break;
            }
            n = tree.parent[n] as usize;
        }
        assert_eq!(covered, 1, "葉 {i} 被 {covered} 個祖先代表");
    }
}

const POSES: [(Vec3A, Vec3A); 3] = [
    (Vec3A::new(10.0, 2.0, -30.0), Vec3A::new(0.2, 0.0, 1.0)),
    (Vec3A::new(-40.0, 1.5, 12.0), Vec3A::new(1.0, 0.0, -0.3)),
    (Vec3A::new(3.0, 30.0, 3.0), Vec3A::new(0.0, -1.0, 0.1)),
];

/// 基準本體:256 頁 × 65536 = 16.7M 節點(= 桌機 page 池的容量,`splats` 陣列 268MB,隨機存取的
/// cache 行為才像真的)、預算 2.5M(將軍府桌機走路階段的飽和情境)、三個姿態、取 `rounds` 次的最小值。
/// 原生與 wasm32-wasip1 跑同一份(`rust/traverse-bench`)。
pub(crate) fn bench_main(rounds: usize) {
    use std::time::Instant;
    let t = Instant::now();
    let tree = build_synth(7, 256 * CHUNK, 256.0);
    let (splats, c2p, root_page) = page_out(&tree);
    eprintln!(
        "synth: {} nodes, {} chunks, built in {:?}",
        tree.nodes.len(), tree.num_chunks, t.elapsed()
    );
    let mut best = [f64::INFINITY; 3];
    for round in 0..rounds {
        for (k, &(origin, forward)) in POSES.iter().enumerate() {
            let p = params(origin, forward);
            let t = Instant::now();
            let (cut, n) = run_atomic(&splats, &c2p, root_page, p, 2_500_000);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            best[k] = best[k].min(ms);
            eprintln!(
                "round {round} pose {k}: {ms:7.1} ms  cut {n}  hash {:#018x}",
                cut_hash(&cut)
            );
        }
    }
    // 取最小值:量的是演算法成本,不是這台機器當下的雜訊
    eprintln!(
        "MIN of {rounds}: pose0 {:.1} ms  pose1 {:.1} ms  pose2 {:.1} ms  sum {:.1} ms",
        best[0], best[1], best[2], best.iter().sum::<f64>()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小樹(~300k 節點、預算 100k)在 debug 下也跑得動:每個姿態的 cut 指紋釘死。
    /// **改迴圈的任何一項,這三個常數都不能動**(動了 = 集合變了 = 那不是微優化)。
    #[test]
    fn synth_cut_hash_is_stable() {
        let tree = build_synth(42, 300_000, 256.0);
        let (splats, c2p, root_page) = page_out(&tree);
        assert!(tree.num_chunks >= 4, "要跨多個 chunk 才測得到 page 映射:{}", tree.num_chunks);
        let expected: [(u64, usize); 3] = [
            (0xf36811e1f2f5514c, 99_996),
            (0x4e1237e27b505c9a, 99_998),
            (0xe5e6b8c91a8fea8b, 99_997),
        ];
        for (k, &(origin, forward)) in POSES.iter().enumerate() {
            let (cut, n) = run_atomic(&splats, &c2p, root_page, params(origin, forward), 100_000);
            assert_valid_cut(&tree, &c2p, &cut);
            assert_eq!(n, cut.len());
            let h = cut_hash(&cut);
            assert_eq!(
                (h, n), expected[k],
                "pose {k}:cut 指紋變了(hash={h:#x}, n={n})。若是刻意改了生成器才允許更新常數。"
            );
        }
    }

    /// 原生基準:`cargo test --release --lib bench_synth -- --ignored --nocapture`;
    /// wasm 基準走 `rust/traverse-bench`(同一份 `bench_main`)。
    #[test]
    #[ignore]
    fn bench_synth_traverse() {
        bench_main(5);
    }
}

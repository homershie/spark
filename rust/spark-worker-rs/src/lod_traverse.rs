//! LoD traverse 的純 Rust 核心:best-first 展開迴圈、可中斷、狀態可跨呼叫保留。
//!
//! 不碰 js_sys / wasm_bindgen,所以原生 `cargo test` 跑得動。JS 邊界(參數解析、
//! `Uint32Array` 打包、`Date::now()`)留在 lod_tree.rs。
//!
//! 不變式:任何時刻 `output ∪ frontier` 都是一份合法 cut(整棵樹被完整覆蓋、無重疊),
//! 且 best-first 保證先展開的是螢幕上最大的節點 —— 所以中途停下輸出,得到的是
//! 「最該細的地方已經細了」的粗 cut。這是時間切片(spec §4)的立足點。

use std::collections::BinaryHeap;

use ahash::AHashSet;
use glam::Vec3A;

use crate::lod_splat::LodSplat;

/// `chunk_to_page` 裡「這個 chunk 不在 GPU」的標記(與 lod_tree.rs 的 0xFFFFFFFF 同值)。
pub(crate) const NOT_RESIDENT: u32 = 0xFFFF_FFFF;

/// 一個 instance 的姿態與 foveation 參數。原子路徑(`run_atomic`)每次呼叫重算一次;
/// 增量路徑(`IncrementalCut::tick`)拿它跟自己存的上一次比對,判斷姿態變了要不要歸零
/// `visited_since_change`(見 lod_cut.rs)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InstanceParams {
    pub(crate) origin: Vec3A,
    pub(crate) forward: Vec3A,
    pub(crate) lod_scale: f32,
    pub(crate) behind_foveate: f32,
    pub(crate) cone_foveate: f32,
    pub(crate) cone_dot0: f32,
    pub(crate) cone_dot: f32,
}

/// 迴圈需要的每-instance 唯讀視圖。`splats` 以 paged_index(page << 16 | offset)索引。
pub(crate) struct TreeView<'a> {
    pub(crate) lod_id: u32,
    pub(crate) splats: &'a [LodSplat],
    pub(crate) chunk_to_page: &'a [u32],
    pub(crate) params: InstanceParams,
}

/// frontier heap 的一筆:`[pixel_scale 的 f32 bits:32 | inst:7 | paged_index:25]` 打包成一個 u64。
///
/// 方案 C(2026-09-11)唯一量得出差別的微優化:原本是 `(OrderedFloat<f32>, u32, u32)`
/// 12 bytes、三段 derive 比較 + NaN 分支,sift-down 每層比兩次、走 21 層;改成 u64 後
/// 一次整數比較、heap 1.8M 筆時 21.6MB → 14.4MB。原生基準 1085 → 568ms(1.91×)。
///
/// 排序與原 tuple **逐位相同**(所以 cut 集合不變):非負 f32 的 bit pattern 單調遞增,
/// 再依 inst、paged 決勝。前提是 key ≥ 0 —— `new()` 把負值 / NaN 夾到 0(pixel_scale 由
/// size × foveate / distance 算出,只有 foveate 設成負數才會負;NaN 只會來自壞資料,
/// 夾到 0 = 永不展開,比原本 OrderedFloat 把 NaN 排最大、優先展開合理)。
///
/// 位寬:inst ≤ [`MAX_INSTANCES`],paged_index < [`MAX_PAGED_INDEX`](= 512 頁 × 65536);
/// `lod_tree.rs` 在 `traverse_lod_trees` 入口驗,超過回 Err,不會靜默錯位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Entry(u64);

pub(crate) const INST_BITS: u32 = 7;
pub(crate) const PAGED_BITS: u32 = 25;
pub(crate) const MAX_INSTANCES: usize = 1 << INST_BITS;
pub(crate) const MAX_PAGED_INDEX: usize = 1 << PAGED_BITS;
const PAGED_MASK: u64 = (1 << PAGED_BITS) - 1;
const INST_MASK: u64 = (1 << INST_BITS) - 1;

impl Entry {
    #[inline]
    pub(crate) fn new(key: f32, inst: u32, paged: u32) -> Self {
        debug_assert!((inst as usize) < MAX_INSTANCES && (paged as usize) < MAX_PAGED_INDEX);
        let key = key.max(0.0); // 負值與 NaN 都落到 +0(f32::max 對 NaN 回另一邊,單指令)
        Self(((key.to_bits() as u64) << (INST_BITS + PAGED_BITS)) | ((inst as u64) << PAGED_BITS) | (paged as u64 & PAGED_MASK))
    }
    #[inline]
    pub(crate) fn key(self) -> f32 {
        f32::from_bits((self.0 >> (INST_BITS + PAGED_BITS)) as u32)
    }
    #[inline]
    pub(crate) fn inst(self) -> u32 {
        ((self.0 >> PAGED_BITS) & INST_MASK) as u32
    }
    #[inline]
    pub(crate) fn paged(self) -> u32 {
        (self.0 & PAGED_MASK) as u32
    }
}

/// 跨呼叫保留的 traverse 狀態。
#[derive(Debug, Default)]
pub(crate) struct TraverseCore {
    pub(crate) frontier: BinaryHeap<Entry>,
    pub(crate) output: Vec<(u32, u32)>,
    pub(crate) touched: Vec<(u32, u32)>,
    pub(crate) touched_set: AHashSet<(u32, u32)>,
    pub(crate) num_splats: usize,
    pub(crate) min_pixel_scale: f32,
    pub(crate) leaf_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopExit {
    /// 既有三個結束條件之一:frontier 空 / 最大節點已小於門檻 / 預算滿。
    Done,
    /// `should_stop` 回 true;狀態原封不動,再叫一次 `expand_until` 就續跑。
    Paused,
}

impl TraverseCore {
    pub(crate) fn reset(&mut self, max_splats: usize) {
        self.frontier.clear();
        self.output.clear();
        self.output.reserve(max_splats);
        self.touched.clear();
        self.touched_set.clear();
        self.num_splats = 0;
        self.min_pixel_scale = f32::INFINITY;
        self.leaf_count = 0;
    }

    fn touch(&mut self, key: (u32, u32)) {
        if self.touched_set.insert(key) {
            self.touched.push(key);
        }
    }

    /// output ∪ frontier。**不 drain**——原子路徑(`run_atomic`)一律跑到 `Done` 才讀,heap 屆時
    /// 已空,`snapshot()` 純粹是把 `output` 攤成 cut;不 drain 的設計留給 bench 與既有測試
    /// (`lod_traverse::tests`)驗證「暫停快照仍是合法 cut」時複用同一顆 core 續跑。
    pub(crate) fn snapshot(&self) -> Vec<(u32, u32)> {
        let mut cut = Vec::with_capacity(self.output.len() + self.frontier.len());
        cut.extend_from_slice(&self.output);
        cut.extend(self.frontier.iter().map(|e| (e.inst(), e.paged())));
        cut
    }
}

/// 把各 instance 的 root 放進 frontier(等同既有迴圈前的那段)。
pub(crate) fn seed_roots(core: &mut TraverseCore, trees: &[TreeView], root_pages: &[u32]) {
    for (inst_index, tree) in trees.iter().enumerate() {
        let root_page = root_pages[inst_index];
        let root_page = if root_page == NOT_RESIDENT { 0 } else { root_page };
        let root_index = root_page << 16;
        let pixel_scale = compute_pixel_scale(&tree.splats[root_index as usize], &tree.params);
        core.frontier.push(Entry::new(pixel_scale, inst_index as u32, root_index));
        core.num_splats += 1;
        core.touch((tree.lod_id, 0));
    }
}

/// best-first 展開。**迴圈本體與 v2.1.0 的 `traverse_lod_trees` 一字不差**,只多了
/// 每次迭代開頭的 `should_stop` 檢查點。
pub(crate) fn expand_until(
    core: &mut TraverseCore,
    trees: &[TreeView],
    max_splats: usize,
    pixel_scale_limit: f32,
    should_stop: &mut impl FnMut() -> bool,
) -> LoopExit {
    while let Some(&top) = core.frontier.peek() {
        let (pixel_scale, inst_index, paged_index) = (top.key(), top.inst(), top.paged());
        if should_stop() {
            return LoopExit::Paused;
        }
        core.min_pixel_scale = core.min_pixel_scale.min(pixel_scale);
        if pixel_scale <= pixel_scale_limit {
            return LoopExit::Done;
        }

        let tree = &trees[inst_index as usize];
        let LodSplat { child_count, child_start, .. } = tree.splats[paged_index as usize];

        if child_count == 0 {
            core.frontier.pop();
            core.output.push((inst_index, paged_index));
            core.leaf_count += 1;
            continue;
        }

        let new_num_splats = core.num_splats - 1 + child_count as usize;
        if new_num_splats > max_splats {
            return LoopExit::Done;
        }

        core.frontier.pop();

        let first_chunk = child_start >> 16;
        core.touch((tree.lod_id, first_chunk));
        let last_chunk = (child_start + child_count as u32 - 1) >> 16;
        if last_chunk != first_chunk {
            core.touch((tree.lod_id, last_chunk));
        }

        if last_chunk as usize >= tree.chunk_to_page.len() {
            core.output.push((inst_index, paged_index));
            continue;
        }
        let first_page = tree.chunk_to_page[first_chunk as usize];
        let last_page = tree.chunk_to_page[last_chunk as usize];
        if first_page == NOT_RESIDENT || last_page == NOT_RESIDENT {
            core.output.push((inst_index, paged_index));
            continue;
        }

        for child in child_start..child_start + child_count as u32 {
            let child_chunk = (child >> 16) as usize;
            let child_page = tree.chunk_to_page[child_chunk];
            let paged = (child_page << 16) | (child & 0xffff);
            let ps = compute_pixel_scale(&tree.splats[paged as usize], &tree.params);
            if ps <= pixel_scale_limit {
                core.output.push((inst_index, paged));
            } else {
                core.frontier.push(Entry::new(ps, inst_index, paged));
            }
        }

        core.num_splats = new_num_splats;
    }
    LoopExit::Done
}

/// 把「像素尺度門檻」換成 heap key 的空間。今天 key 就是 pixel_scale,所以是恆等;
/// 這層存在是讓 key 的表示(例如改成平方)只改一處。
pub(crate) fn limit_key(pixel_scale_limit: f32) -> f32 {
    pixel_scale_limit
}

/// 與 v2.1.0 的 `compute_pixel_scale` 同式,只是參數改吃 `InstanceParams`。
pub(crate) fn compute_pixel_scale(splat: &LodSplat, p: &InstanceParams) -> f32 {
    compute_pixel_scale_raw(splat.center(), splat.size(), p)
}

/// 同上,吃已解出的 center/size(IncrementalCut 的 arena 存 f16 副本,不經過 LodSplat)。
pub(crate) fn compute_pixel_scale_raw(center: Vec3A, size: f32, p: &InstanceParams) -> f32 {
    let delta = center - p.origin;
    let distance = delta.length().max(1.0e-6);
    let inv_distance = 1.0 / distance;
    let pixel_scale = size * inv_distance * p.lod_scale;
    let forward_dot = delta.dot(p.forward);
    let foveate = if forward_dot <= 0.0 {
        p.behind_foveate
    } else {
        let dot = forward_dot * inv_distance;
        if dot >= p.cone_dot0 {
            1.0
        } else if dot >= p.cone_dot {
            let t = (dot - p.cone_dot) / (p.cone_dot0 - p.cone_dot);
            p.cone_foveate + (1.0 - p.cone_foveate) * t
        } else {
            let t = dot / p.cone_dot;
            p.behind_foveate + (p.cone_foveate - p.behind_foveate) * t
        }
    };
    foveate * pixel_scale
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ahash::AHashMap;
    use glam::Vec3;
    use ordered_float::OrderedFloat;

    const LOD_ID: u32 = 7;
    const LEAF_FIRST: u32 = 21;
    const LEAF_LAST: u32 = 84;

    /// 三層四叉樹,全部在 chunk 0(paged_index == index):
    /// root(0, size 8) → 1..=4(size 4) → 5..=20(size 2) → 21..=84 葉(size 1)。
    /// 回傳 (splats, parent) —— parent[root] = u32::MAX。
    pub(crate) fn build_tree() -> (Vec<LodSplat>, Vec<u32>) {
        let mut splats = vec![LodSplat::default(); 85];
        let mut parent = vec![u32::MAX; 85];
        splats[0] = LodSplat::new(Vec3::ZERO, 8.0, 1, 4);
        for i in 1..=4usize {
            splats[i] = LodSplat::new(Vec3::new(i as f32, 0.0, 0.0), 4.0, (5 + (i - 1) * 4) as u32, 4);
            parent[i] = 0;
        }
        for i in 5..=20usize {
            splats[i] = LodSplat::new(Vec3::new(i as f32, 1.0, 0.0), 2.0, (21 + (i - 5) * 4) as u32, 4);
            parent[i] = (1 + (i - 5) / 4) as u32;
        }
        for i in 21..=84usize {
            splats[i] = LodSplat::new(Vec3::new(i as f32 * 0.1, 2.0, 0.0), 1.0, 0, 0);
            parent[i] = (5 + (i - 21) / 4) as u32;
        }
        (splats, parent)
    }

    /// foveation 全 1(cone_dot0 = cone_dot = 1 → 任何角度都走最後一支、算出 1),
    /// 相機在 z = -50,節點在 z ≈ 0 → pixel_scale ≈ size / 50,層與層之間有明確大小差。
    pub(crate) fn params() -> InstanceParams {
        InstanceParams {
            origin: Vec3A::new(0.0, 0.0, -50.0),
            forward: Vec3A::Z,
            lod_scale: 1.0,
            behind_foveate: 1.0,
            cone_foveate: 1.0,
            cone_dot0: 1.0,
            cone_dot: 1.0,
        }
    }

    fn view<'a>(splats: &'a [LodSplat], chunk_to_page: &'a [u32]) -> Vec<TreeView<'a>> {
        vec![TreeView { lod_id: LOD_ID, splats, chunk_to_page, params: params() }]
    }

    /// 跑到 Done;`stop_every` = 每次呼叫 `expand_until` 跑滿幾次迭代就暫停(0 = 不暫停)。
    /// 回傳每次暫停時的快照 + 最終快照。⚠️ 是「跑滿 N 次才停」不是「第 N 次開頭就停」——
    /// 後者在 N = 1 時會在做任何事之前就回 Paused,永遠不前進。
    fn run(core: &mut TraverseCore, trees: &[TreeView], max_splats: usize, stop_every: u32) -> Vec<Vec<(u32, u32)>> {
        core.reset(max_splats);
        seed_roots(core, trees, &[0]);
        let mut snaps = Vec::new();
        loop {
            let mut n = 0u32;
            let exit = expand_until(core, trees, max_splats, 0.0, &mut || {
                n += 1;
                stop_every > 0 && n > stop_every
            });
            snaps.push(core.snapshot());
            if exit == LoopExit::Done {
                return snaps;
            }
        }
    }

    fn sorted(cut: &[(u32, u32)]) -> Vec<u32> {
        let mut v: Vec<u32> = cut.iter().map(|&(_, p)| p).collect();
        v.sort_unstable();
        v
    }

    fn assert_valid_cut(cut: &[(u32, u32)], parent: &[u32]) {
        let set: AHashSet<u32> = cut.iter().map(|&(_, p)| p).collect();
        assert_eq!(set.len(), cut.len(), "cut 內有重複節點");
        for leaf in LEAF_FIRST..=LEAF_LAST {
            let mut n = leaf;
            let mut covered = 0;
            loop {
                if set.contains(&n) {
                    covered += 1;
                }
                if parent[n as usize] == u32::MAX {
                    break;
                }
                n = parent[n as usize];
            }
            assert_eq!(covered, 1, "葉 {leaf} 被 {covered} 個祖先代表(應恰 1)");
        }
    }

    #[test]
    fn atomic_run_reaches_all_leaves_when_budget_allows() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = view(&splats, &c2p);
        let mut core = TraverseCore::default();
        let snaps = run(&mut core, &trees, 100, 0);
        assert_eq!(snaps.len(), 1);
        assert_valid_cut(&snaps[0], &parent);
        assert_eq!(sorted(&snaps[0]), (LEAF_FIRST..=LEAF_LAST).collect::<Vec<_>>());
        assert_eq!(core.leaf_count, 64);
        assert_eq!(core.num_splats, 64);
        assert_eq!(core.touched, vec![(LOD_ID, 0)]);
    }

    #[test]
    fn sliced_run_equals_atomic() {
        let (splats, _) = build_tree();
        let c2p = [0u32];
        let trees = view(&splats, &c2p);
        let mut core = TraverseCore::default();
        let atomic = sorted(run(&mut core, &trees, 100, 0).last().unwrap());
        for stop_every in [1, 2, 3, 7, 1000] {
            let snaps = run(&mut core, &trees, 100, stop_every);
            assert_eq!(sorted(snaps.last().unwrap()), atomic, "stop_every={stop_every}");
        }
    }

    /// 原子呼叫(raycast)走另一顆 core:切片中的 `core` 暫停 → 另一顆 core 跑原子 →
    /// `core` 續跑到 Done,最終 cut 必須等於沒被打斷的原子結果(Task 5 review #1 的修法)。
    #[test]
    fn atomic_run_on_scratch_core_does_not_disturb_in_flight_round() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = view(&splats, &c2p);
        let max = 100;

        let mut reference = TraverseCore::default();
        let expected = sorted(run(&mut reference, &trees, max, 0).last().unwrap());

        // 切片中的 round:跑滿 3 次迭代就暫停
        let mut core = TraverseCore::default();
        core.reset(max);
        seed_roots(&mut core, &trees, &[0]);
        let mut n = 0u32;
        let exit = expand_until(&mut core, &trees, max, 0.0, &mut || {
            n += 1;
            n > 3
        });
        assert_eq!(exit, LoopExit::Paused);
        let paused_snapshot = core.snapshot();
        assert_valid_cut(&paused_snapshot, &parent);
        assert!(paused_snapshot.len() < 64, "暫停時應該還沒展到全部葉子");
        let paused_frontier = core.frontier.len();
        let paused_output = core.output.len();

        // 旁路的原子 traverse,在自己的 scratch core 上(不同 max_splats,故意跟 round 不一樣)
        let mut scratch = TraverseCore::default();
        let atomic = run(&mut scratch, &trees, 30, 0);
        assert_eq!(atomic.len(), 1);
        assert!(atomic[0].len() <= 30);

        // core 原封不動
        assert_eq!(core.frontier.len(), paused_frontier);
        assert_eq!(core.output.len(), paused_output);
        assert_eq!(sorted(&core.snapshot()), sorted(&paused_snapshot));

        // 續跑到底,結果等於沒被打斷
        let exit = expand_until(&mut core, &trees, max, 0.0, &mut || false);
        assert_eq!(exit, LoopExit::Done);
        assert_eq!(sorted(&core.snapshot()), expected);
        assert_eq!(core.leaf_count, 64);
    }

    #[test]
    fn every_pause_snapshot_is_valid_cut_and_refines_monotonically() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = view(&splats, &c2p);
        let mut core = TraverseCore::default();
        let snaps = run(&mut core, &trees, 100, 1);
        assert!(snaps.len() > 10, "每次迭代都暫停,應該有很多片");
        for (k, snap) in snaps.iter().enumerate() {
            assert_valid_cut(snap, &parent);
            if k > 0 {
                let prev: AHashSet<u32> = snaps[k - 1].iter().map(|&(_, p)| p).collect();
                for &(_, node) in snap {
                    // 新出現的節點,其 parent 必在上一片(= 只會把節點換成它的 children,不會回退)
                    if !prev.contains(&node) {
                        assert!(prev.contains(&parent[node as usize]), "片 {k}:節點 {node} 憑空出現");
                    }
                }
            }
        }
    }

    #[test]
    fn budget_is_respected_at_every_pause() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = view(&splats, &c2p);
        let mut core = TraverseCore::default();
        let max = 30;
        let snaps = run(&mut core, &trees, max, 1);
        for snap in &snaps {
            assert_valid_cut(snap, &parent);
            assert!(snap.len() <= max, "cut {} > 預算 {max}", snap.len());
        }
        let last = snaps.last().unwrap();
        // 每次展開 +3(4 子 −1 父),1 → 4 → … → 28;再展開一次會到 31 > 30 就停
        assert!(last.len() > max - 4 && last.len() <= max, "最終 cut {}", last.len());
        assert_eq!(core.num_splats, last.len());
    }

    #[test]
    fn non_resident_children_keep_parent_in_output_and_touch_chunk() {
        let (mut splats, parent) = build_tree();
        // 節點 20 的 children 改指到 chunk 1(index 65536..),而 chunk 1 不在 GPU
        splats[20] = LodSplat::new(Vec3::new(20.0, 1.0, 0.0), 2.0, 65536, 4);
        let c2p = [0u32, NOT_RESIDENT];
        let trees = view(&splats, &c2p);
        let mut core = TraverseCore::default();
        let snaps = run(&mut core, &trees, 100, 0);
        let last = snaps.last().unwrap();
        // 20 的 4 片葉(原 81..=84)不再可達;cut 應含 20 本身,不含任何 ≥ 65536 的 paged_index
        let set: AHashSet<u32> = last.iter().map(|&(_, p)| p).collect();
        assert!(set.contains(&20));
        assert!(set.iter().all(|&p| p < 65536));
        assert!(core.touched.contains(&(LOD_ID, 1)), "children 所在 chunk 要進 touched 讓 pager 去抓");
        // 其餘 60 片葉 + 節點 20 = 61
        assert_eq!(last.len(), 61);
        // 合法 cut(把 20 當葉看:81..=84 的祖先鏈上只有 20 在 cut 內)
        let mut parent2 = parent.clone();
        for leaf in 81..=84usize {
            parent2[leaf] = 20;
        }
        assert_valid_cut(last, &parent2);
    }

    /// u64 打包的排序必須與原本的 `(OrderedFloat<f32>, u32, u32)` 逐位相同 —— 這是「集合不變」的前提。
    #[test]
    fn packed_entry_orders_like_the_old_tuple() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut items: Vec<(f32, u32, u32)> = (0..20_000)
            .map(|_| {
                let r = next();
                // key:多數是小正數,少數是 0 / 重複 / inf,逼出並列決勝
                let key = match r % 7 {
                    0 => 0.0,
                    1 => 1.0e-3,
                    2 => f32::INFINITY,
                    _ => ((r >> 8) % 1_000_000) as f32 * 1.0e-9,
                };
                let inst = ((r >> 40) % 3) as u32;
                let paged = ((r >> 20) % 5) as u32 * 65536 + ((r >> 4) % 64) as u32;
                (key, inst, paged)
            })
            .collect();
        items.sort_by_key(|&(k, i, p)| (OrderedFloat(k), i, p));
        let mut packed: Vec<Entry> = items.iter().map(|&(k, i, p)| Entry::new(k, i, p)).collect();
        packed.sort();
        for (a, b) in items.iter().zip(packed.iter()) {
            assert_eq!((a.0, a.1, a.2), (b.key(), b.inst(), b.paged()));
        }
        // 極值:負值與 NaN 落到 +0,inf 保持最大
        assert_eq!(Entry::new(-1.0, 0, 0).key(), 0.0);
        assert_eq!(Entry::new(f32::NAN, 0, 0).key(), 0.0);
        assert!(Entry::new(f32::INFINITY, 0, 0) > Entry::new(f32::MAX, 127, MAX_PAGED_INDEX as u32 - 1));
        let e = Entry::new(0.5, 127, MAX_PAGED_INDEX as u32 - 1);
        assert_eq!((e.key(), e.inst(), e.paged()), (0.5, 127, MAX_PAGED_INDEX as u32 - 1));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 方案 B(真增量)可行性試作(handoff「先做的一件事」)。
    //
    // 要證的一件事:**只在展開節點時順手記 parent**(不改 `.rad`、不建整棵樹的表),
    // 姿態變了之後拿現有 cut 做「太粗的拆、parent 已經夠細的收回」,結果等於用新姿態
    // 從 root 原子跑出來的 cut。
    //
    // 兩個設計上的細節,測試就是為了逼出它們:
    // 1. 收回的判準是「**parent 的 pixel_scale ≤ limit**」,不是「整組兄弟都 ≤ limit」——
    //    原子 traverse 展開一個節點的條件就是它自己的 ps > limit,與孩子各自多大無關
    //    (ps 沿樹不嚴格單調:子節點可以比父節點更靠近相機)。
    // 2. 收回可以連跳兩層以上(葉 → 中層 → 上層),所以 parent 要記在**每一個展開過的
    //    節點**上(`parent_of`),不只 cut 那一筆 —— 中層節點正是我們自己展開出來的,
    //    展開當下就知道它的 parent。
    //
    // 這裡是試作,單 instance、單 chunk、不管 residency / 預算再平衡的細節;
    // 正式的資料結構由 spec 決定。
    // ─────────────────────────────────────────────────────────────────────────

    const NONE: u32 = u32::MAX;

    /// 帶 parent 的 cut。`cut` 的每筆 = (paged_index, parent);`parent_of` = 每個
    /// **展開過**(interior)的節點的 parent。
    #[derive(Debug, Default, Clone)]
    struct ParentedCut {
        cut: Vec<(u32, u32)>,
        parent_of: AHashMap<u32, u32>,
    }

    impl ParentedCut {
        fn nodes(&self) -> Vec<u32> {
            let mut v: Vec<u32> = self.cut.iter().map(|&(n, _)| n).collect();
            v.sort_unstable();
            v
        }
    }

    /// 從 root 原子展開(best-first,與 `expand_until` 同語意),但每展開一個節點就記
    /// `parent_of[child] = node`。等同「今天的 traverse + 順手記 parent」。
    fn expand_recording(splats: &[LodSplat], p: &InstanceParams, limit: f32, max: usize) -> ParentedCut {
        let mut heap: BinaryHeap<(OrderedFloat<f32>, u32, u32)> = BinaryHeap::new();
        let mut out = ParentedCut::default();
        let mut n = 1usize;
        heap.push((OrderedFloat(compute_pixel_scale(&splats[0], p)), 0, NONE));
        while let Some(&(OrderedFloat(ps), node, parent)) = heap.peek() {
            if ps <= limit {
                break;
            }
            let LodSplat { child_count, child_start, .. } = splats[node as usize];
            if child_count == 0 {
                heap.pop();
                out.cut.push((node, parent));
                continue;
            }
            if n - 1 + child_count as usize > max {
                break;
            }
            heap.pop();
            out.parent_of.insert(node, parent);
            for c in child_start..child_start + child_count as u32 {
                heap.push((OrderedFloat(compute_pixel_scale(&splats[c as usize], p)), c, node));
            }
            n = n - 1 + child_count as usize;
        }
        out.cut.extend(heap.into_iter().map(|(_, node, parent)| (node, parent)));
        out
    }

    /// 姿態變了:在現有 cut 上做局部收回 / 展開,不從 root 重來。
    ///
    /// 1. 限制收回:interior 節點 P 的 ps ≤ limit 且**它的孩子全部直接在 cut 裡** → 收回成 P。
    ///    反覆做到沒有為止(收回後 P 進 cut,P 的 parent 可能接著符合)。
    /// 2. 預算收回:|cut| > max → 從 ps 最小的「孩子全在 cut」的 interior 開始收,收到 ≤ max。
    /// 3. 展開:cut 內 ps > limit 且有孩子的節點,依 ps 由大到小展開、裝得下才展(= 原子的
    ///    greedy 停法);展開出來的孩子照樣進 heap。
    fn rebalance(inc: &mut ParentedCut, splats: &[LodSplat], p: &InstanceParams, limit: f32, max: usize) {
        let ps = |n: u32| compute_pixel_scale(&splats[n as usize], p);

        // 「孩子全在 cut」的 interior 集合:cut 內以 P 為 parent 的筆數 == P 的 child_count
        let complete_groups = |inc: &ParentedCut| -> Vec<u32> {
            let mut count: AHashMap<u32, u16> = AHashMap::new();
            for &(_, parent) in &inc.cut {
                if parent != NONE {
                    *count.entry(parent).or_default() += 1;
                }
            }
            count.into_iter()
                .filter(|&(parent, c)| c == splats[parent as usize].child_count)
                .map(|(parent, _)| parent)
                .collect()
        };
        let collapse = |inc: &mut ParentedCut, parent: u32| {
            inc.cut.retain(|&(_, pp)| pp != parent);
            let grand = inc.parent_of.remove(&parent).expect("interior 節點必有記錄的 parent");
            inc.cut.push((parent, grand));
        };

        // 1. 限制收回(到不動點)
        loop {
            let victims: Vec<u32> = complete_groups(inc).into_iter().filter(|&g| ps(g) <= limit).collect();
            if victims.is_empty() {
                break;
            }
            for g in victims {
                collapse(inc, g);
            }
        }
        // 2. 預算收回(從 ps 最小的完整組開始)
        while inc.cut.len() > max {
            let mut groups = complete_groups(inc);
            groups.sort_by_key(|&g| OrderedFloat(ps(g)));
            let g = *groups.first().expect("超過預算卻沒有可收的組");
            collapse(inc, g);
        }
        // 3. 展開(best-first,greedy)
        let mut heap: BinaryHeap<(OrderedFloat<f32>, u32, u32)> = inc.cut.iter()
            .filter(|&&(n, _)| splats[n as usize].child_count > 0)
            .map(|&(n, parent)| (OrderedFloat(ps(n)), n, parent))
            .filter(|&(OrderedFloat(s), _, _)| s > limit)
            .collect();
        let mut n = inc.cut.len();
        while let Some(&(_, node, parent)) = heap.peek() {
            let LodSplat { child_count, child_start, .. } = splats[node as usize];
            if n - 1 + child_count as usize > max {
                break;
            }
            heap.pop();
            inc.cut.retain(|&(m, _)| m != node);
            inc.parent_of.insert(node, parent);
            for c in child_start..child_start + child_count as u32 {
                inc.cut.push((c, node));
                let s = ps(c);
                if s > limit && splats[c as usize].child_count > 0 {
                    heap.push((OrderedFloat(s), c, node));
                }
            }
            n = n - 1 + child_count as usize;
        }
    }

    fn pose(origin: Vec3A) -> InstanceParams {
        InstanceParams { origin, ..params() }
    }

    /// interior 的 parent 記錄必須與 cut 自洽:cut 每筆的 parent(非 root)都在 `parent_of`,
    /// 且沿 parent 鏈走得到 root。
    fn assert_parent_chain(inc: &ParentedCut, truth: &[u32]) {
        for &(node, parent) in &inc.cut {
            assert_eq!(parent, truth[node as usize], "節點 {node} 記錄的 parent 錯");
            let mut p = parent;
            while p != NONE {
                assert_eq!(inc.parent_of.get(&p).copied(), Some(truth[p as usize]), "interior {p} 的 parent 記錄缺或錯");
                p = truth[p as usize];
            }
        }
    }

    /// 樹的 ps(z = -50):root .16 / L1 .08 / L2 .04 / 葉 .02。limit 0.03:
    ///   z=-50  → 展開到 L2(.04 > .03)→ cut = 64 葉
    ///   z=-100 → root .08 / L1 .04 / L2 .02 → cut = 16 個 L2
    ///   z=-150 → root .053 / L1 .027 → cut = 4 個 L1(要從葉連收兩層)
    #[test]
    fn incremental_coarsen_two_levels_equals_atomic() {
        let (splats, truth) = build_tree();
        let limit = 0.03;
        let near = pose(Vec3A::new(0.0, 0.0, -50.0));
        let far = pose(Vec3A::new(0.0, 0.0, -150.0));

        let mut inc = expand_recording(&splats, &near, limit, 1000);
        assert_eq!(inc.nodes(), (LEAF_FIRST..=LEAF_LAST).collect::<Vec<_>>());
        assert_parent_chain(&inc, &truth);

        rebalance(&mut inc, &splats, &far, limit, 1000);
        let atomic = expand_recording(&splats, &far, limit, 1000);
        assert_eq!(inc.nodes(), vec![1, 2, 3, 4], "拉遠後應收回到 L1");
        assert_eq!(inc.nodes(), atomic.nodes());
        assert_parent_chain(&inc, &truth);
        assert_valid_cut(&inc.cut.iter().map(|&(n, _)| (0, n)).collect::<Vec<_>>(), &truth);
    }

    /// 混合 cut:limit 0.03999 卡在 z=-100 時 L1 的 ps(4/√(x²+100²),x=1..4 → .039998/.039992/
    /// .039982/.039968)中間 —— 節點 1、2 展開、3、4 不展;相機 x 偏到 5 則反過來(4、3 展)。
    /// 所以同一層「有的拆、有的收」,連同拉近(全葉)/ 拉遠(全 L1)串成一條鏈,每站都要等於原子。
    #[test]
    fn incremental_refine_and_mixed_chain_equals_atomic() {
        let (splats, truth) = build_tree();
        let limit = 0.03999;
        let chain = [
            Vec3A::new(0.0, 0.0, -100.0), // 混合:1、2 展
            Vec3A::new(0.0, 0.0, -50.0),  // 全葉
            Vec3A::new(5.0, 0.0, -100.0), // 混合:4、3 展(從全葉收成混合)
            Vec3A::new(0.0, 0.0, -150.0), // 全 L1
            Vec3A::new(0.0, 0.0, -100.0), // 混合:1、2 展(從 L1 拆成混合)
            Vec3A::new(5.0, 0.0, -100.0), // 混合 → 另一種混合(同時有拆有收)
            Vec3A::new(0.0, 0.0, -50.0),  // 全葉
        ];
        let mut inc = expand_recording(&splats, &pose(chain[0]), limit, 1000);
        assert_eq!(inc.nodes(), vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "起點應是混合 cut");
        let mut distinct = AHashSet::new();
        distinct.insert(inc.nodes());
        for &origin in &chain[1..] {
            let p = pose(origin);
            rebalance(&mut inc, &splats, &p, limit, 1000);
            let atomic = expand_recording(&splats, &p, limit, 1000);
            assert_eq!(inc.nodes(), atomic.nodes(), "origin={origin:?}");
            assert_parent_chain(&inc, &truth);
            assert_valid_cut(&inc.cut.iter().map(|&(n, _)| (0, n)).collect::<Vec<_>>(), &truth);
            distinct.insert(inc.nodes());
        }
        assert_eq!(distinct.len(), 4, "全葉 / 全 L1 / 兩種混合 = 4 種 cut");
    }

    /// 預算:64 葉 → max 30 要收 12 組;原子在同一姿態下展開 ps 最大的 4 個 L2,
    /// 剩 12 個沒展 = 我們收掉的 12 個最小的。反向(從 4 個 L1 展到 30)也要相等。
    /// ⚠️ 這個相等靠的是本樹每次展開都 +3(child_count 一律 4):原子的「第一個裝不下就停」
    /// 與增量的「裝得下才展」在等步長之下才是同一件事;不等步長的差異留給 spec 討論。
    #[test]
    fn incremental_budget_rebalance_equals_atomic() {
        let (splats, truth) = build_tree();
        let limit = 0.03;
        let near = pose(Vec3A::new(3.0, 0.0, -50.0)); // 側偏一點,讓 L2 之間 ps 有序可分
        let far = pose(Vec3A::new(0.0, 0.0, -150.0));
        let max = 30;

        // 無預算的 64 葉 → 收到 30
        let mut inc = expand_recording(&splats, &near, limit, 1000);
        rebalance(&mut inc, &splats, &near, limit, max);
        let atomic = expand_recording(&splats, &near, limit, max);
        assert_eq!(inc.cut.len(), 28);
        assert_eq!(inc.nodes(), atomic.nodes());
        assert_parent_chain(&inc, &truth);

        // 4 個 L1 → 展到 30
        let mut inc = expand_recording(&splats, &far, limit, 1000);
        assert_eq!(inc.cut.len(), 4);
        rebalance(&mut inc, &splats, &near, limit, max);
        assert_eq!(inc.nodes(), atomic.nodes());
        assert_parent_chain(&inc, &truth);
    }
}

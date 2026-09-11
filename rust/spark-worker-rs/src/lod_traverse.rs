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
use ordered_float::OrderedFloat;

use crate::lod_tree::LodSplat;

/// `chunk_to_page` 裡「這個 chunk 不在 GPU」的標記(與 lod_tree.rs 的 0xFFFFFFFF 同值)。
pub(crate) const NOT_RESIDENT: u32 = 0xFFFF_FFFF;

/// 一個 instance 的姿態與 foveation 參數。round 開始時算一次、續跑沿用。
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

/// 跨呼叫保留的 traverse 狀態。
#[derive(Debug, Default)]
pub(crate) struct TraverseCore {
    pub(crate) frontier: BinaryHeap<(OrderedFloat<f32>, u32, u32)>,
    pub(crate) output: Vec<(u32, u32)>,
    pub(crate) touched: Vec<(u32, u32)>,
    pub(crate) touched_set: AHashSet<(u32, u32)>,
    pub(crate) num_splats: usize,
    pub(crate) min_pixel_scale: f32,
    pub(crate) leaf_count: u32,
}

/// 一輪(round)的固定參數:從「姿態變了、從 root 重開」到「跑完」。
#[derive(Debug, Clone)]
pub(crate) struct RoundMeta {
    pub(crate) lod_ids: Vec<u32>,
    pub(crate) params: Vec<InstanceParams>,
    pub(crate) max_splats: usize,
    pub(crate) pixel_scale_limit: f32,
    /// 已跑過幾片(從 0 起;`traverse_lod_trees` 每跑完一片 +1)。
    pub(crate) slice: u32,
    pub(crate) done: bool,
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

    /// output ∪ frontier。**不 drain** —— heap 留給下一片續跑(spec D3)。
    pub(crate) fn snapshot(&self) -> Vec<(u32, u32)> {
        let mut cut = Vec::with_capacity(self.output.len() + self.frontier.len());
        cut.extend_from_slice(&self.output);
        cut.extend(self.frontier.iter().map(|&(_, inst, paged)| (inst, paged)));
        cut
    }
}

/// 這一輪要不要從 root 重開。`lod_ids` 不同 = instance 集合變了,`inst_index` 對不上,視同重開。
pub(crate) fn is_fresh(atomic: bool, restart: bool, round: Option<&RoundMeta>, lod_ids: &[u32]) -> bool {
    atomic
        || restart
        || match round {
            None => true,
            Some(r) => r.done || r.lod_ids.as_slice() != lod_ids,
        }
}

/// 把各 instance 的 root 放進 frontier(等同既有迴圈前的那段)。
pub(crate) fn seed_roots(core: &mut TraverseCore, trees: &[TreeView], root_pages: &[u32]) {
    for (inst_index, tree) in trees.iter().enumerate() {
        let root_page = root_pages[inst_index];
        let root_page = if root_page == NOT_RESIDENT { 0 } else { root_page };
        let root_index = root_page << 16;
        let pixel_scale = compute_pixel_scale(&tree.splats[root_index as usize], &tree.params);
        core.frontier.push((OrderedFloat(pixel_scale), inst_index as u32, root_index));
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
    while let Some(&(OrderedFloat(pixel_scale), inst_index, paged_index)) = core.frontier.peek() {
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
                core.frontier.push((OrderedFloat(ps), inst_index, paged));
            }
        }

        core.num_splats = new_num_splats;
    }
    LoopExit::Done
}

/// 與 v2.1.0 的 `compute_pixel_scale` 同式,只是參數改吃 `InstanceParams`。
pub(crate) fn compute_pixel_scale(splat: &LodSplat, p: &InstanceParams) -> f32 {
    let center = splat.center();
    let delta = center - p.origin;
    let distance = delta.length().max(1.0e-6);
    let inv_distance = 1.0 / distance;
    let pixel_scale = splat.size() * inv_distance * p.lod_scale;

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
mod tests {
    use super::*;
    use glam::Vec3;

    const LOD_ID: u32 = 7;
    const LEAF_FIRST: u32 = 21;
    const LEAF_LAST: u32 = 84;

    /// 三層四叉樹,全部在 chunk 0(paged_index == index):
    /// root(0, size 8) → 1..=4(size 4) → 5..=20(size 2) → 21..=84 葉(size 1)。
    /// 回傳 (splats, parent) —— parent[root] = u32::MAX。
    fn build_tree() -> (Vec<LodSplat>, Vec<u32>) {
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
    fn params() -> InstanceParams {
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

    #[test]
    fn is_fresh_rules() {
        let meta = |done: bool, ids: &[u32]| RoundMeta {
            lod_ids: ids.to_vec(),
            params: vec![],
            max_splats: 1,
            pixel_scale_limit: 0.0,
            slice: 1,
            done,
        };
        assert!(is_fresh(true, false, Some(&meta(false, &[1])), &[1]), "atomic 一律重開");
        assert!(is_fresh(false, true, Some(&meta(false, &[1])), &[1]), "restart 一律重開");
        assert!(is_fresh(false, false, None, &[1]), "沒有 round 就重開");
        assert!(is_fresh(false, false, Some(&meta(true, &[1])), &[1]), "上一輪已完成就重開");
        assert!(is_fresh(false, false, Some(&meta(false, &[1])), &[1, 2]), "instance 集合變了就重開");
        assert!(!is_fresh(false, false, Some(&meta(false, &[1, 2])), &[1, 2]), "同一輪未完成 → 續跑");
    }
}

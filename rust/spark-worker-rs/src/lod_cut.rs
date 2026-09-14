//! 方案 B:真增量的 LoD cut。設計與 17 條決策見本專案
//! `docs/superpowers/specs/2026-09-11-incremental-traverse-design.md`(§4)。
//!
//! cut 不直接存「哪些節點在畫面上」,而是存**展開過的節點**(`Node`,interior)與每個 interior 的
//! 孩子 `arena`;cut = 每個 interior 沒被展開的孩子。表裡存 **chunk-space 索引**(`child_start`
//! 那個空間),paged index 只在 `pack` 時查 `chunk_to_page` 現算 —— 所以 pager 把槽位換成別的
//! chunk 不會讓表失效,只會在 pack 時露出成 `evicted`(spec D3)。
//!
//! 純 Rust、不碰 js_sys;deadline 用注入的閉包,原生 `cargo test` 跑得動。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use ahash::AHashMap;
use glam::Vec3A;
use half::f16;
use ordered_float::OrderedFloat;

use crate::lod_splat::LodSplat;
use crate::lod_traverse::{compute_pixel_scale_raw, InstanceParams, TreeView, NOT_RESIDENT};

pub(crate) const NONE: u32 = u32::MAX;
/// arena:沒展開、還有孩子可拆。
pub(crate) const CUT_LEAF: u32 = u32::MAX - 1;
/// arena:沒展開、樹葉(不能拆;收由 parent 決定)。
pub(crate) const CUT_TERMINAL: u32 = u32::MAX - 2;
/// arena free list 的分桶上限(child_count ≤ 32 進桶;更大的 bump 配置、釋放只計數,spec D16)。
const ARENA_BUCKETS: usize = 33;
/// 掃描 / 拆時每幾筆查一次 deadline(A 的 D4)。
const DEADLINE_STRIDE: u32 = 4096;

/// 展開過的節點。`child_count == 0` 且 `parent == NONE` 且不是 root ⇒ 已釋放(在 `node_free`)。
#[derive(Debug, Clone, Copy)]
pub(crate) struct Node {
    pub(crate) inst: u8,
    /// chunk-space 索引;虛擬根 = NONE。
    pub(crate) index: u32,
    pub(crate) parent: u32,
    pub(crate) child_start: u32,
    pub(crate) child_count: u16,
    pub(crate) expanded: u16,
    pub(crate) ps: f32,
    pub(crate) children_base: u32,
    pub(crate) center: [f16; 3],
    pub(crate) size: f16,
    /// 孩子 center 離自己 center 的最大距離、孩子 size 的最大值(展開時算):掃描用來算整組孩子的
    /// ps 保守上界,上界 ≤ 拆門檻就整組跳過(spec D18)。
    pub(crate) radius: f16,
    pub(crate) child_size_max: f16,
    /// pack 發現孩子頁被踢 → 下一 tick 無視 ps 收回(spec §4.4)。
    pub(crate) forced: bool,
    pub(crate) alive: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Child {
    /// 展開了 → 該孩子的 Node slot;否則 CUT_LEAF / CUT_TERMINAL。
    pub(crate) slot: u32,
    pub(crate) center: [f16; 3],
    pub(crate) size: f16,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ExpandOutcome {
    Expanded(u32),
    /// 孩子(或它的孩子)所在 chunk 不在 GPU;已記進 `wanted`。
    NotResident,
    /// 這個孩子是樹葉。
    Terminal,
    /// 拆了會超過 max_splats;帶 child_count 讓呼叫端算門檻。
    Misfit(u16),
}

pub(crate) struct Packed {
    /// 每 instance 一份 paged index(未補齊 16384 倍數;lod_tree.rs 打包時補)。
    pub(crate) indices: Vec<Vec<u32>>,
    pub(crate) evicted: u32,
}

pub(crate) struct Chunks {
    /// `(lod_id, chunk)`,順序 = fetchPriority:roots、needed、wanted(ps 由大到小)。
    pub(crate) list: Vec<(u32, u32)>,
    pub(crate) roots: usize,
    pub(crate) needed: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct TickStats {
    pub(crate) scanned: u32,
    pub(crate) expanded: u32,
    pub(crate) collapsed: u32,
    pub(crate) passes: u32,
    pub(crate) settled: bool,
    /// `tick()` 本身不 pack,不填這個欄位(Task 4 接 `pack()` 之後才會有值)。
    #[allow(dead_code)]
    pub(crate) evicted: u32,
    pub(crate) wanted_count: u32,
    pub(crate) cut_size: u32,
    pub(crate) t: f32,
    pub(crate) arena_leaked: u32,
    pub(crate) changed: bool,
}

pub(crate) struct IncrementalCut {
    pub(crate) nodes: Vec<Node>,
    node_free: Vec<u32>,
    pub(crate) arena: Vec<Child>,
    arena_free: Vec<Vec<u32>>,
    arena_leaked: u32,
    pub(crate) roots: Vec<u32>,
    pub(crate) cut_size: usize,
    /// 每 instance:chunk → 住在這個 chunk 的 cut 孩子數(needed 的來源)。
    pub(crate) chunk_refs: Vec<AHashMap<u32, u32>>,
    /// (inst, chunk) → (想要它的最大 ps = fetch 優先序, 最後一次要它的 pass)。兩個 pass 沒再要就從
    /// `chunks()` 消失(相機走開了,不該一直叫 pager 抓沒用的頁)。
    pub(crate) wanted: AHashMap<(u8, u32), (f32, u32)>,
    pub(crate) lod_ids: Vec<u32>,
    pub(crate) max_splats: usize,
    pub(crate) limit: f32,
    pub(crate) t: f32,
    pub(crate) generation: u64,
    packed_generation: u64,
    cursor: usize,
    /// 上次 tick 的姿態 / 參數;變了就把「這一 pass 已掃過的」歸零(settled 要求變動後掃完一整 pass)。
    params: Vec<InstanceParams>,
    visited_since_change: usize,
    /// 這一 pass 開始時的 generation;pass 走完沒變 = clean pass(settled 的條件之一)。
    pass_start_generation: u64,
    clean_pass: bool,
    passes: u32,
    /// (ps, node slot, child i, node.index 驗證用)—— 跨 tick 保留,pop 時驗證。
    expand_heap: BinaryHeap<(OrderedFloat<f32>, u32, u16, u32)>,
    collapse_heap: BinaryHeap<Reverse<(OrderedFloat<f32>, u32, u32)>>,
    forced: Vec<u32>,
    /// 掃描 / 拆時每幾筆查一次 deadline(A 的 D4);預設 `DEADLINE_STRIDE`,production 兩處
    /// (掃描、拆)共用同一個量級。測試用:設成 1 讓注入的 deadline 閉包真的能在任意一步中斷,
    /// 驗證「切到一半仍是合法 cut」的路徑。
    pub(crate) deadline_stride: u32,
}

impl IncrementalCut {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_free: Vec::new(),
            arena: Vec::new(),
            arena_free: (0..ARENA_BUCKETS).map(|_| Vec::new()).collect(),
            arena_leaked: 0,
            roots: Vec::new(),
            cut_size: 0,
            chunk_refs: Vec::new(),
            wanted: AHashMap::new(),
            lod_ids: Vec::new(),
            max_splats: 0,
            limit: 0.0,
            t: 0.0,
            generation: 0,
            packed_generation: u64::MAX,
            cursor: 0,
            params: Vec::new(),
            visited_since_change: 0,
            pass_start_generation: 0,
            clean_pass: false,
            passes: 0,
            expand_heap: BinaryHeap::new(),
            collapse_heap: BinaryHeap::new(),
            forced: Vec::new(),
            deadline_stride: DEADLINE_STRIDE,
        }
    }

    // ── 樹的存取 ────────────────────────────────────────────────────────────

    /// chunk-space 索引 → 節點(resident 才有)。
    fn splat_at<'a>(tree: &'a TreeView<'a>, index: u32) -> Option<&'a LodSplat> {
        let chunk = (index >> 16) as usize;
        if chunk >= tree.chunk_to_page.len() {
            return None;
        }
        let page = tree.chunk_to_page[chunk];
        if page == NOT_RESIDENT {
            return None;
        }
        Some(&tree.splats[((page << 16) | (index & 0xffff)) as usize])
    }

    fn resident(tree: &TreeView, chunk: u32) -> bool {
        (chunk as usize) < tree.chunk_to_page.len() && tree.chunk_to_page[chunk as usize] != NOT_RESIDENT
    }

    fn ps_of(center: [f16; 3], size: f16, p: &InstanceParams) -> f32 {
        compute_pixel_scale_raw(Vec3A::from_array(center.map(|x| x.to_f32())), size.to_f32(), p)
    }

    fn want(&mut self, inst: u8, chunk: u32, ps: f32) {
        let pass = self.passes;
        self.wanted.entry((inst, chunk)).and_modify(|v| { v.0 = v.0.max(ps); v.1 = pass; }).or_insert((ps, pass));
    }

    // ── 配置 ───────────────────────────────────────────────────────────────

    fn alloc_node(&mut self, node: Node) -> u32 {
        if let Some(slot) = self.node_free.pop() {
            self.nodes[slot as usize] = node;
            slot
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as u32
        }
    }

    fn free_node(&mut self, slot: u32) {
        let n = &mut self.nodes[slot as usize];
        n.alive = false;
        n.child_count = 0;
        n.expanded = 0;
        n.parent = NONE;
        self.node_free.push(slot);
    }

    fn alloc_arena(&mut self, count: u16) -> u32 {
        let c = count as usize;
        if c < ARENA_BUCKETS {
            if let Some(base) = self.arena_free[c].pop() {
                return base;
            }
        }
        let base = self.arena.len() as u32;
        self.arena.resize(self.arena.len() + c, Child { slot: CUT_TERMINAL, center: [f16::ZERO; 3], size: f16::ZERO });
        base
    }

    fn free_arena(&mut self, base: u32, count: u16) {
        let c = count as usize;
        if c < ARENA_BUCKETS {
            self.arena_free[c].push(base);
        } else {
            self.arena_leaked += count as u32;
        }
    }

    fn refs_add(&mut self, inst: u8, chunk: u32, delta: i32) {
        let e = self.chunk_refs[inst as usize].entry(chunk).or_insert(0);
        let v = *e as i64 + delta as i64;
        debug_assert!(v >= 0, "chunk refcount 變負");
        if v <= 0 {
            self.chunk_refs[inst as usize].remove(&chunk);
        } else {
            *e = v as u32;
        }
    }

    /// 一段連續 chunk-space 索引 `start..start+count` 住在哪些 chunk、各幾個(≤ 2 段)。
    fn chunk_spans(start: u32, count: u16) -> [(u32, i32); 2] {
        let first = start >> 16;
        let last = (start + count as u32 - 1) >> 16;
        if first == last {
            [(first, count as i32), (NONE, 0)]
        } else {
            let in_first = ((first + 1) << 16) - start;
            [(first, in_first as i32), (last, count as i32 - in_first as i32)]
        }
    }

    // ── restart ─────────────────────────────────────────────────────────────

    /// 清表、每個 instance 種一個虛擬根(child_start = root 的 chunk-space 索引 0,child_count = 1)。
    pub(crate) fn restart(&mut self, trees: &[TreeView], root_pages: &[u32], limit: f32) {
        self.nodes.clear();
        self.node_free.clear();
        self.arena.clear();
        for b in &mut self.arena_free {
            b.clear();
        }
        self.arena_leaked = 0;
        self.roots.clear();
        self.cut_size = 0;
        self.chunk_refs = trees.iter().map(|_| AHashMap::new()).collect();
        self.wanted.clear();
        self.lod_ids = trees.iter().map(|t| t.lod_id).collect();
        self.params = trees.iter().map(|t| t.params).collect();
        self.visited_since_change = 0;
        self.limit = limit;
        self.t = limit;
        self.generation += 1;
        self.cursor = 0;
        self.pass_start_generation = self.generation;
        self.clean_pass = false;
        self.expand_heap.clear();
        self.collapse_heap.clear();
        self.forced.clear();

        for (inst, tree) in trees.iter().enumerate() {
            // v2.1.0 慣例:root page 未知就當 page 0(seed_roots 同)
            let root_page = if root_pages[inst] == NOT_RESIDENT { 0 } else { root_pages[inst] };
            let root = &tree.splats[(root_page << 16) as usize];
            let base = self.alloc_arena(1);
            self.arena[base as usize] = Child {
                slot: if root.child_count == 0 { CUT_TERMINAL } else { CUT_LEAF },
                center: root.center,
                size: root.size,
            };
            let slot = self.alloc_node(Node {
                inst: inst as u8,
                index: NONE,
                parent: NONE,
                child_start: 0,
                child_count: 1,
                expanded: 0,
                ps: f32::INFINITY,
                children_base: base,
                center: root.center,
                size: root.size,
                radius: f16::ZERO,
                child_size_max: root.size,
                forced: false,
                alive: true,
            });
            self.roots.push(slot);
            self.refs_add(inst as u8, 0, 1);
            self.cut_size += 1;
        }
    }

    // ── expand / collapse ───────────────────────────────────────────────────

    /// 把 `slot` 的第 `i` 個孩子(必須是 CUT_LEAF 或 CUT_TERMINAL)拆成它的孩子。`ps` 是呼叫端算好的該孩子 ps
    /// (成為新 Node 的 `ps`,也是 `wanted` 的優先序)。
    pub(crate) fn expand(&mut self, trees: &[TreeView], slot: u32, i: u16, ps: f32, max_splats: usize) -> ExpandOutcome {
        let node = self.nodes[slot as usize];
        debug_assert!(node.alive && i < node.child_count);
        let inst = node.inst;
        let tree = &trees[inst as usize];
        let child_index = node.child_start + i as u32;
        let arena_i = (node.children_base + i as u32) as usize;
        // 孩子在建立時已依 child_count == 0 預分類(見上面 restart / expand 的填 arena 迴圈),
        // 所以這裡可能已經是 CUT_TERMINAL(例如剛好是葉);唯一不合法的是已經展開過
        // (arena 存的是真的 Node slot)。
        debug_assert!(
            matches!(self.arena[arena_i].slot, CUT_LEAF | CUT_TERMINAL),
            "expand 只能作用在還沒展開的孩子(CUT_LEAF/CUT_TERMINAL),而不是已展開的 Node slot"
        );

        let Some(child) = Self::splat_at(tree, child_index) else {
            self.want(inst, child_index >> 16, ps);
            return ExpandOutcome::NotResident;
        };
        let LodSplat { child_count, child_start, center, size } = child.clone();
        if child_count == 0 {
            self.arena[arena_i].slot = CUT_TERMINAL;
            return ExpandOutcome::Terminal;
        }
        if self.cut_size + child_count as usize - 1 > max_splats {
            return ExpandOutcome::Misfit(child_count);
        }
        let spans = Self::chunk_spans(child_start, child_count);
        for &(chunk, n) in &spans {
            if n > 0 && !Self::resident(tree, chunk) {
                self.want(inst, chunk, ps);
                return ExpandOutcome::NotResident;
            }
        }

        let base = self.alloc_arena(child_count);
        let my_center = Vec3A::from_array(center.map(|x| x.to_f32()));
        let mut radius = 0.0f32;
        let mut child_size_max = 0.0f32;
        for g in 0..child_count as u32 {
            let s = Self::splat_at(tree, child_start + g).expect("首尾 chunk 都 resident,中間不可能不 resident");
            radius = radius.max((s.center() - my_center).length());
            child_size_max = child_size_max.max(s.size());
            self.arena[(base + g) as usize] = Child {
                slot: if s.child_count == 0 { CUT_TERMINAL } else { CUT_LEAF },
                center: s.center,
                size: s.size,
            };
        }
        // f16 是向下取整的風險:上界要保守,往上取一格
        let radius = f16::from_f32(radius * 1.001 + 1e-3);
        let child_size_max = f16::from_f32(child_size_max * 1.001);
        let new_slot = self.alloc_node(Node {
            inst,
            index: child_index,
            parent: slot,
            child_start,
            child_count,
            expanded: 0,
            ps,
            children_base: base,
            center,
            size,
            radius,
            child_size_max,
            forced: false,
            alive: true,
        });
        self.arena[arena_i].slot = new_slot;
        self.nodes[slot as usize].expanded += 1;
        for &(chunk, n) in &spans {
            if n > 0 {
                self.refs_add(inst, chunk, n);
            }
        }
        self.refs_add(inst, child_index >> 16, -1);
        self.wanted.remove(&(inst, child_index >> 16));
        for &(chunk, n) in &spans {
            if n > 0 {
                self.wanted.remove(&(inst, chunk));
            }
        }
        self.cut_size += child_count as usize - 1;
        self.generation += 1;
        ExpandOutcome::Expanded(new_slot)
    }

    /// 把 `slot` 這個 interior 的整組孩子收回成它自己。要求:不是虛擬根、`expanded == 0`、
    /// 自己的 chunk resident(否則記 `wanted`、留孩子)。
    pub(crate) fn collapse(&mut self, trees: &[TreeView], slot: u32) -> bool {
        let node = self.nodes[slot as usize];
        if !node.alive || node.parent == NONE || node.expanded != 0 {
            return false;
        }
        let inst = node.inst;
        let tree = &trees[inst as usize];
        if !Self::resident(tree, node.index >> 16) {
            self.want(inst, node.index >> 16, node.ps);
            return false;
        }
        // parent 的 arena 裡找到自己
        let parent = self.nodes[node.parent as usize];
        let mut my_i = None;
        for i in 0..parent.child_count as u32 {
            if self.arena[(parent.children_base + i) as usize].slot == slot {
                my_i = Some(i);
                break;
            }
        }
        let my_i = my_i.expect("parent 的 arena 必含自己");
        self.arena[(parent.children_base + my_i) as usize].slot = CUT_LEAF;
        self.nodes[node.parent as usize].expanded -= 1;
        for &(chunk, n) in &Self::chunk_spans(node.child_start, node.child_count) {
            if n > 0 {
                self.refs_add(inst, chunk, -n);
            }
        }
        self.refs_add(inst, node.index >> 16, 1);
        self.wanted.remove(&(inst, node.index >> 16));
        self.free_arena(node.children_base, node.child_count);
        self.free_node(slot);
        self.cut_size -= node.child_count as usize - 1;
        self.generation += 1;
        true
    }

    // ── pack / chunks ───────────────────────────────────────────────────────

    /// 遍歷表輸出 cut 的 paged index。孩子的 chunk 不 resident → 不輸出(洞)、`evicted += 1`、
    /// 其 parent 進 `forced`(下一 tick 無視 ps 收回)。
    pub(crate) fn pack(&mut self, trees: &[TreeView]) -> Packed {
        let mut indices: Vec<Vec<u32>> = trees.iter().map(|_| Vec::new()).collect();
        let mut evicted = 0u32;
        for slot in 0..self.nodes.len() {
            let node = self.nodes[slot];
            if !node.alive {
                continue;
            }
            let tree = &trees[node.inst as usize];
            for i in 0..node.child_count as u32 {
                let c = self.arena[(node.children_base + i) as usize];
                if c.slot != CUT_LEAF && c.slot != CUT_TERMINAL {
                    continue;
                }
                let index = node.child_start + i;
                let chunk = index >> 16;
                if !Self::resident(tree, chunk) {
                    evicted += 1;
                    if !self.nodes[slot].forced {
                        self.nodes[slot].forced = true;
                        self.forced.push(slot as u32);
                    }
                    continue;
                }
                let page = tree.chunk_to_page[chunk as usize];
                indices[node.inst as usize].push((page << 16) | (index & 0xffff));
            }
        }
        self.packed_generation = self.generation;
        Packed { indices, evicted }
    }

    pub(crate) fn needs_pack(&self) -> bool {
        self.packed_generation != self.generation
    }

    /// fetchPriority:每 instance 的 root chunk、needed(refcount > 0,chunk 遞增)、wanted(ps 遞減)。
    pub(crate) fn chunks(&self) -> Chunks {
        let mut list = Vec::new();
        for &id in &self.lod_ids {
            list.push((id, 0));
        }
        let roots = list.len();
        for (inst, refs) in self.chunk_refs.iter().enumerate() {
            let mut cs: Vec<u32> = refs.iter().filter(|(_, &n)| n > 0).map(|(&c, _)| c).collect();
            cs.sort_unstable();
            for c in cs {
                if c != 0 {
                    list.push((self.lod_ids[inst], c));
                }
            }
        }
        let needed = list.len() - roots;
        // 兩個 pass 沒再要的不列(相機走開了)
        let fresh = self.passes.saturating_sub(1);
        let mut w: Vec<_> = self.wanted.iter().filter(|(_, v)| v.1 >= fresh).collect();
        w.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(b.0)));
        for (&(inst, chunk), _) in w {
            let id = self.lod_ids[inst as usize];
            if !list.contains(&(id, chunk)) {
                list.push((id, chunk));
            }
        }
        Chunks { list, roots, needed }
    }

    // ── tick / settled ──────────────────────────────────────────────────────

    /// 穩 = 參數變動之後掃完過一整 pass、最近一整 pass 沒有拆收、沒有候選、預算內。
    /// ⚠️ `wanted` 非空**不**阻止 settled:頁到了 JS 會因 `lodTreeDirty` 再 tick,頁沒到 tick 也沒事做。
    pub(crate) fn settled(&self) -> bool {
        self.visited_since_change >= self.nodes.len() && self.clean_pass
            && self.expand_heap.is_empty() && self.collapse_heap.is_empty()
            && self.forced.is_empty() && self.cut_size <= self.max_splats
    }

    /// 一次 worker 呼叫的工作(spec §4.2)。掃描階段跑到 `scan_deadline`(預算的 60%,免得掃描把
    /// 時間吃光、拆收永遠輪不到),其餘階段跑到 `deadline`;任一回 true 就停,狀態永遠合法。
    /// 介面由 spec §4.8 / task brief 指定,參數不可減——同 `lod_tree.rs` 的 `traverse_lod_trees` 先例。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick(
        &mut self, trees: &[TreeView], root_pages: &[u32], max_splats: usize, limit: f32, eps: f32,
        scan_deadline: &mut impl FnMut() -> bool, deadline: &mut impl FnMut() -> bool,
    ) -> TickStats {
        let ids: Vec<u32> = trees.iter().map(|t| t.lod_id).collect();
        if ids != self.lod_ids || self.nodes.is_empty() {
            self.restart(trees, root_pages, limit);
        }
        let params: Vec<InstanceParams> = trees.iter().map(|t| t.params).collect();
        if params != self.params || max_splats != self.max_splats || limit != self.limit {
            self.params = params;
            self.visited_since_change = 0;
            self.clean_pass = false;
        }
        self.max_splats = max_splats;
        self.limit = limit;
        if self.t < limit {
            self.t = limit;
        }
        let gen0 = self.generation;
        let mut st = TickStats::default();
        let up = self.t * (1.0 + eps);
        let down = self.t * (1.0 - eps);
        let limit_down = limit * (1.0 - eps);

        // ── 1. 掃描切片(最多一整 pass)──
        // over_budget:超預算時,收候選不能只靠 ps 掉到門檻以下才進 heap(D20「超預算從 ps 最小的收」)——
        // 否則相機不動、每個節點的 ps 都穩穩高於門檻,收 heap 永遠是空的,`max_splats` 被外部調低後永遠收不回來
        // (下面消費端雖有 `over` 快速通關,但沒有候選可 pop 一樣無效)。這裡用 tick 開始時的 cut_size 判斷一次即可:
        // 掃描階段不改動 cut_size,直到步驟 2/3 才會變。
        let over_budget = self.cut_size > self.max_splats;
        let n = self.nodes.len();
        let mut checks = 0u32;
        let mut stopped = false;
        for _ in 0..n {
            if self.cursor >= n {
                self.cursor = 0;
                self.passes += 1;
                self.clean_pass = self.generation == self.pass_start_generation;
                self.pass_start_generation = self.generation;
            }
            let slot = self.cursor as u32;
            self.cursor += 1;
            self.visited_since_change += 1;
            checks += 1;
            if checks % self.deadline_stride == 0 && scan_deadline() {
                break; // 掃描配額用完;拆收仍有自己的時間
            }
            let node = self.nodes[slot as usize];
            if !node.alive {
                continue;
            }
            st.scanned += 1;
            let p = &trees[node.inst as usize].params;
            let node_ps = if node.parent == NONE { f32::INFINITY } else { Self::ps_of(node.center, node.size, p) };
            self.nodes[slot as usize].ps = node_ps;
            // 2a. 階層上界(spec D18):整組孩子的 ps 不可能超過
            //     lod_scale · child_size_max / max(dist − radius, ε)(foveate ≤ 1),
            //     ≤ 拆門檻就整組跳過 —— 遠場幾乎全中,掃描省 5/6 的算術。虛擬根不跳(radius 0、只有 root)。
            let base = node.children_base as usize;
            let skip_children = if node.parent == NONE {
                false
            } else {
                let d = (Vec3A::from_array(node.center.map(|x| x.to_f32())) - p.origin).length();
                let ps_max = p.lod_scale * node.child_size_max.to_f32() / (d - node.radius.to_f32()).max(1.0e-6);
                ps_max <= up
            };
            if !skip_children {
                for i in 0..node.child_count as usize {
                    let c = self.arena[base + i];
                    if c.slot != CUT_LEAF {
                        continue;
                    }
                    let ps = Self::ps_of(c.center, c.size, p);
                    if ps > up {
                        self.expand_heap.push((OrderedFloat(ps), slot, i as u16, node.index));
                    }
                }
            }
            if node.parent != NONE && node.expanded == 0 && (node_ps <= down || node_ps <= limit_down || over_budget) {
                self.collapse_heap.push(Reverse((OrderedFloat(node_ps), slot, node.index)));
            }
        }
        // 掃完整 pass 但 pass 內有變動 → clean_pass 已由上面設;若這 tick 一路掃到 n 又沒 wrap
        // (cursor == n),下一 tick 開頭會 wrap 並結算。

        // ── 2. 收:forced(頁被踢)無條件;候選在「超預算」或「ps ≤ limit」時 ──
        let forced = std::mem::take(&mut self.forced);
        for slot in forced {
            let node = self.nodes[slot as usize];
            if node.alive && node.forced {
                self.nodes[slot as usize].forced = false;
                if self.collapse(trees, slot) {
                    st.collapsed += 1;
                } else {
                    // 收不了(parent 不 resident 或有孩子展開):留著,下次 pack 再標
                }
            }
        }
        let mut last_collapsed_ps: Option<f32> = None;
        while let Some(&Reverse((OrderedFloat(ps), slot, index))) = self.collapse_heap.peek() {
            let over = self.cut_size > self.max_splats;
            if !(over || ps <= limit_down) {
                self.collapse_heap.clear(); // 剩下的 ps 都更大、`over` 對它們同樣不成立(同 expand_heap 的對稱處理)
                break;
            }
            self.collapse_heap.pop();
            let node = self.nodes[slot as usize];
            if !node.alive || node.index != index || node.expanded != 0 {
                continue; // lazy:候選過期
            }
            if self.collapse(trees, slot) {
                st.collapsed += 1;
                if over {
                    last_collapsed_ps = Some(ps);
                }
                // 連收多層:parent 可能因此成為候選
                let parent = self.nodes[node.parent as usize];
                if parent.parent != NONE && parent.expanded == 0 {
                    let pps = Self::ps_of(parent.center, parent.size, &trees[parent.inst as usize].params);
                    self.nodes[node.parent as usize].ps = pps;
                    if pps <= down || pps <= limit_down {
                        self.collapse_heap.push(Reverse((OrderedFloat(pps), node.parent, parent.index)));
                    }
                }
            }
            if deadline() {
                stopped = true;
                break;
            }
        }
        if let Some(ps) = last_collapsed_ps {
            self.t = self.t.max(ps);
        }

        // ── 3. 拆:ps 由大到小,裝得下、resident、時間沒到 ──
        let mut misfit_ps: Option<f32> = None;
        let mut expansions = 0u32;
        if !stopped {
            let up_now = self.t * (1.0 + eps);
            while let Some(&(OrderedFloat(ps), slot, i, index)) = self.expand_heap.peek() {
                if ps <= up_now {
                    self.expand_heap.clear(); // 剩下的都更小
                    break;
                }
                self.expand_heap.pop();
                let node = self.nodes[slot as usize];
                if !node.alive || node.index != index || self.arena[(node.children_base + i as u32) as usize].slot != CUT_LEAF {
                    continue; // lazy:候選過期
                }
                match self.expand(trees, slot, i, ps, self.max_splats) {
                    ExpandOutcome::Expanded(new_slot) => {
                        st.expanded += 1;
                        let nn = self.nodes[new_slot as usize];
                        let p = &trees[nn.inst as usize].params;
                        for g in 0..nn.child_count as usize {
                            let c = self.arena[nn.children_base as usize + g];
                            if c.slot == CUT_LEAF {
                                let cps = Self::ps_of(c.center, c.size, p);
                                if cps > up_now {
                                    self.expand_heap.push((OrderedFloat(cps), new_slot, g as u16, nn.index));
                                }
                            }
                        }
                    }
                    ExpandOutcome::Misfit(_) => {
                        misfit_ps = Some(ps);
                        self.expand_heap.clear();
                        break;
                    }
                    ExpandOutcome::NotResident | ExpandOutcome::Terminal => {}
                }
                expansions += 1;
                if expansions % self.deadline_stride == 0 && deadline() {
                    stopped = true;
                    break;
                }
            }
        }

        // ── 4. 門檻控制器(spec §4.5)──
        if let Some(ps) = misfit_ps {
            self.t = ps;
        } else if !stopped && self.expand_heap.is_empty() && self.t > limit
            && (self.cut_size as f32) < 0.98 * self.max_splats as f32
        {
            self.t = (self.t * 0.9).max(limit);
        }
        if self.t < limit {
            self.t = limit;
        }

        st.passes = self.passes;
        st.settled = self.settled();
        st.wanted_count = self.wanted.len() as u32;
        st.cut_size = self.cut_size as u32;
        st.t = self.t;
        st.arena_leaked = self.arena_leaked;
        st.changed = self.generation != gen0;
        st
    }

    /// 測試用:cut 的 (inst, chunk-space index) 集合。
    #[cfg(test)]
    pub(crate) fn cut_set_for_test(&self) -> Vec<(u8, u32)> {
        let mut out = Vec::new();
        for node in &self.nodes {
            if !node.alive {
                continue;
            }
            for i in 0..node.child_count as u32 {
                let c = self.arena[(node.children_base + i) as usize];
                if c.slot == CUT_LEAF || c.slot == CUT_TERMINAL {
                    out.push((node.inst, node.child_start + i));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lod_traverse::tests::{build_tree, params};
    use ahash::AHashSet;

    const LOD_ID: u32 = 7;

    fn views<'a>(splats: &'a [LodSplat], c2p: &'a [u32]) -> Vec<TreeView<'a>> {
        vec![TreeView { lod_id: LOD_ID, splats, chunk_to_page: c2p, params: params() }]
    }

    /// cut 的 chunk-space 索引集合(排序),測試用。
    fn cut_indices(cut: &IncrementalCut) -> Vec<u32> {
        let mut v: Vec<u32> = cut.cut_set_for_test().into_iter().map(|(_, idx)| idx).collect();
        v.sort_unstable();
        v
    }

    fn assert_valid(cut: &IncrementalCut, parent: &[u32]) {
        let set: AHashSet<u32> = cut_indices(cut).into_iter().collect();
        assert_eq!(set.len(), cut.cut_size, "cut_size 帳與實際不符");
        for leaf in 21..=84u32 {
            let (mut n, mut covered) = (leaf, 0);
            loop {
                if set.contains(&n) { covered += 1; }
                if parent[n as usize] == u32::MAX { break; }
                n = parent[n as usize];
            }
            assert_eq!(covered, 1, "葉 {leaf} 被 {covered} 個祖先代表");
        }
    }

    #[test]
    fn restart_seeds_root_as_single_cut_leaf() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = views(&splats, &c2p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.0);
        assert_eq!(cut_indices(&cut), vec![0]);
        assert_eq!(cut.cut_size, 1);
        assert_valid(&cut, &parent);
        assert_eq!(cut.chunk_refs[0].get(&0).copied(), Some(1));
        // pack:root 的 paged index = (page 0 << 16) | 0
        let packed = cut.pack(&trees);
        assert_eq!(packed.indices[0], vec![0]);
        assert_eq!(packed.evicted, 0);
    }

    #[test]
    fn expand_then_collapse_round_trips() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let trees = views(&splats, &c2p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.0);
        let root_slot = cut.roots[0];
        // 拆 root(虛擬根的第 0 個孩子)
        let out = cut.expand(&trees, root_slot, 0, 1.0, 1000);
        assert!(matches!(out, ExpandOutcome::Expanded(_)));
        assert_eq!(cut_indices(&cut), vec![1, 2, 3, 4]);
        assert_eq!(cut.cut_size, 4);
        assert_valid(&cut, &parent);
        // 再拆節點 2(root Node 的第 1 個孩子)
        let ExpandOutcome::Expanded(node0) = out else { unreachable!() };
        let out = cut.expand(&trees, node0, 1, 1.0, 1000);
        let ExpandOutcome::Expanded(node2) = out else { panic!("{out:?}") };
        assert_eq!(cut_indices(&cut), vec![1, 3, 4, 9, 10, 11, 12]);
        assert_valid(&cut, &parent);
        assert_eq!(cut.nodes[node0 as usize].expanded, 1);
        // 收回節點 2
        assert!(cut.collapse(&trees, node2));
        assert_eq!(cut_indices(&cut), vec![1, 2, 3, 4]);
        assert_eq!(cut.nodes[node0 as usize].expanded, 0);
        assert_valid(&cut, &parent);
        // 虛擬根不能收;有孩子展開的不能收
        assert!(!cut.collapse(&trees, cut.roots[0]));
        cut.expand(&trees, node0, 1, 1.0, 1000);
        assert!(!cut.collapse(&trees, node0), "node0 有孩子展開,不能收");
        assert_eq!(cut.generation, 5, "restart +1、拆 ×3、收 ×1");
    }

    #[test]
    fn expand_respects_budget_and_terminal() {
        let (splats, _) = build_tree();
        let c2p = [0u32];
        let trees = views(&splats, &c2p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.0);
        let root_slot = cut.roots[0];
        // 預算 3:root 拆成 4 裝不下
        assert_eq!(cut.expand(&trees, root_slot, 0, 1.0, 3), ExpandOutcome::Misfit(4));
        assert_eq!(cut.cut_size, 1);
        let ExpandOutcome::Expanded(n0) = cut.expand(&trees, root_slot, 0, 1.0, 4) else { panic!() };
        let ExpandOutcome::Expanded(n1) = cut.expand(&trees, n0, 0, 1.0, 100) else { panic!() };
        let ExpandOutcome::Expanded(n5) = cut.expand(&trees, n1, 0, 1.0, 100) else { panic!() };
        // 21..=24 是葉:arena 標 CUT_TERMINAL,再拆回 Terminal
        assert_eq!(cut.expand(&trees, n5, 0, 1.0, 100), ExpandOutcome::Terminal);
        let base = cut.nodes[n5 as usize].children_base as usize;
        assert!(cut.arena[base..base + 4].iter().all(|c| c.slot == CUT_TERMINAL));
    }

    #[test]
    fn chunk_space_index_survives_page_remap() {
        // 整棵樹放在 chunk 0,但 chunk 0 映射到 page 2(非恆等);paged = 2<<16 | index
        let (splats0, parent) = build_tree();
        let mut splats = vec![LodSplat::default(); 3 * 65536];
        for (i, s) in splats0.iter().enumerate() { splats[(2 << 16) + i] = s.clone(); }
        let c2p = [2u32];
        let trees = views(&splats, &c2p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[2], 0.0);
        let r = cut.roots[0];
        let ExpandOutcome::Expanded(n0) = cut.expand(&trees, r, 0, 1.0, 100) else { panic!() };
        cut.expand(&trees, n0, 2, 1.0, 100);
        assert_eq!(cut_indices(&cut), vec![1, 2, 4, 13, 14, 15, 16]);
        assert_valid(&cut, &parent);
        let packed = cut.pack(&trees);
        let mut got = packed.indices[0].clone();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 4, 13, 14, 15, 16].iter().map(|i| (2 << 16) | i).collect::<Vec<u32>>());
        // 頁重映射:chunk 0 現在在 page 1(資料也搬過去),表不用動,pack 出新的 paged
        let mut splats2 = vec![LodSplat::default(); 3 * 65536];
        for (i, s) in splats0.iter().enumerate() { splats2[(1 << 16) + i] = s.clone(); }
        let c2p2 = [1u32];
        let trees2 = views(&splats2, &c2p2);
        let packed = cut.pack(&trees2);
        let mut got = packed.indices[0].clone();
        got.sort_unstable();
        assert_eq!(got, vec![1, 2, 4, 13, 14, 15, 16].iter().map(|i| (1 << 16) | i).collect::<Vec<u32>>());
        assert_eq!(packed.evicted, 0);
    }

    #[test]
    fn chunks_lists_roots_needed_then_wanted() {
        let (mut splats, _) = build_tree();
        // 節點 20 的孩子在 chunk 1,不 resident
        splats[20] = LodSplat::new(glam::Vec3::new(20.0, 1.0, 0.0), 2.0, 65536, 4);
        let c2p = [0u32, NOT_RESIDENT];
        let trees = views(&splats, &c2p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.0);
        let r = cut.roots[0];
        let ExpandOutcome::Expanded(n0) = cut.expand(&trees, r, 0, 1.0, 100) else { panic!() };
        let ExpandOutcome::Expanded(n4) = cut.expand(&trees, n0, 3, 1.0, 100) else { panic!() };
        // 節點 20 = n4 的第 3 個孩子;孩子 chunk 1 不 resident → NotResident + wanted
        assert_eq!(cut.expand(&trees, n4, 3, 0.5, 100), ExpandOutcome::NotResident);
        assert_eq!(cut.wanted.get(&(0, 1)).map(|v| v.0), Some(0.5));
        let chunks = cut.chunks();
        assert_eq!(chunks.roots, 1);
        assert_eq!(chunks.needed, 0, "cut 全在 chunk 0 = root chunk,已在 roots 段、needed 不重複列");
        assert_eq!(chunks.list, vec![(LOD_ID, 0), (LOD_ID, 1)]);
    }

    /// 原子參考:從 root best-first 展開到底(與 expand_until 同語意),回 chunk-space cut。
    fn atomic_cut(splats: &[LodSplat], p: &InstanceParams, limit: f32, max: usize) -> Vec<u32> {
        let mut heap: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        let mut out = Vec::new();
        let mut n = 1usize;
        heap.push((OrderedFloat(crate::lod_traverse::compute_pixel_scale(&splats[0], p)), 0));
        while let Some(&(OrderedFloat(ps), node)) = heap.peek() {
            if ps <= limit { break; }
            let LodSplat { child_count, child_start, .. } = splats[node as usize];
            if child_count == 0 { heap.pop(); out.push(node); continue; }
            if n - 1 + child_count as usize > max { break; }
            heap.pop();
            for c in child_start..child_start + child_count as u32 {
                heap.push((OrderedFloat(crate::lod_traverse::compute_pixel_scale(&splats[c as usize], p)), c));
            }
            n = n - 1 + child_count as usize;
        }
        out.extend(heap.into_iter().map(|(_, node)| node));
        out.sort_unstable();
        out
    }

    fn pose(origin: Vec3A) -> InstanceParams {
        InstanceParams { origin, ..params() }
    }

    /// tick 到 settled(無 deadline),回 tick 數;超過 50 tick 視為不收斂。
    fn settle(cut: &mut IncrementalCut, trees: &[TreeView], max: usize, limit: f32, eps: f32) -> u32 {
        for k in 1..=50 {
            let st = cut.tick(trees, &[0], max, limit, eps, &mut || false, &mut || false);
            if st.settled { return k; }
        }
        panic!("50 tick 未 settled:cut_size {} t {}", cut.cut_size, cut.t);
    }

    fn views_p<'a>(splats: &'a [LodSplat], c2p: &'a [u32], p: InstanceParams) -> Vec<TreeView<'a>> {
        vec![TreeView { lod_id: LOD_ID, splats, chunk_to_page: c2p, params: p }]
    }

    /// 姿態鏈(handoff 試作的三個情境合一):全 L2(站 1、6)/ 全 L1(站 3)/ 兩種混合(站 0、2、4、5);
    /// 每站 == 原子;eps = 0 才逐位相等。limit=0.03999 之下 L2 的 ps 在這條鏈上從未超過門檻
    /// (z=-50 時最大 ≈0.0398),所以鏈上能到達的最細層是 L2、不是樹葉 21..=84(原子參考同樣停在 L2)。
    #[test]
    fn inc_equals_atomic_limit_only() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let limit = 0.03999;
        let chain = [
            Vec3A::new(0.0, 0.0, -100.0), Vec3A::new(0.0, 0.0, -50.0), Vec3A::new(5.0, 0.0, -100.0),
            Vec3A::new(0.0, 0.0, -150.0), Vec3A::new(0.0, 0.0, -100.0), Vec3A::new(5.0, 0.0, -100.0),
            Vec3A::new(0.0, 0.0, -50.0),
        ];
        let mut cut = IncrementalCut::new();
        let mut distinct = AHashSet::new();
        for (k, &o) in chain.iter().enumerate() {
            let p = pose(o);
            let trees = views_p(&splats, &c2p, p);
            if k == 0 { cut.restart(&trees, &[0], limit); }
            let ticks = settle(&mut cut, &trees, 1000, limit, 0.0);
            assert!(ticks <= 6, "站 {k} 用了 {ticks} tick");
            assert_eq!(cut_indices(&cut), atomic_cut(&splats, &p, limit, 1000), "站 {k} origin={o:?}");
            assert_valid(&cut, &parent);
            distinct.insert(cut_indices(&cut));
        }
        assert_eq!(distinct.len(), 4);
        assert_eq!(cut_indices(&cut), (5..=20).collect::<Vec<_>>());
    }

    /// 預算:64 葉 → 30 收 12 組 == 原子;4 個 L1 → 30 拆到 28 == 原子(等步長樹,handoff 註)。
    #[test]
    fn inc_equals_atomic_with_budget() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let limit = 0.03;
        let near = pose(Vec3A::new(3.0, 0.0, -50.0));
        let far = pose(Vec3A::new(0.0, 0.0, -150.0));
        let atomic = atomic_cut(&splats, &near, limit, 30);
        assert_eq!(atomic.len(), 28);

        let trees = views_p(&splats, &c2p, near);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], limit);
        settle(&mut cut, &trees, 1000, limit, 0.0);
        assert_eq!(cut.cut_size, 64);
        settle(&mut cut, &trees, 30, limit, 0.0);          // 預算縮到 30:收
        assert_eq!(cut_indices(&cut), atomic);
        assert_valid(&cut, &parent);

        let trees_far = views_p(&splats, &c2p, far);
        settle(&mut cut, &trees_far, 1000, limit, 0.0);   // 拉遠:4 個 L1
        assert_eq!(cut_indices(&cut), vec![1, 2, 3, 4]);
        settle(&mut cut, &trees, 30, limit, 0.0);         // 拉近 + 預算 30:拆到 28
        assert_eq!(cut_indices(&cut), atomic);
        assert_valid(&cut, &parent);
    }

    /// deadline 每筆就停:每個 tick 之後都是合法 cut、不超預算,最終仍收斂到原子。
    #[test]
    fn inc_every_tick_is_valid_cut() {
        let (splats, parent) = build_tree();
        let c2p = [0u32];
        let limit = 0.03;
        let p = pose(Vec3A::new(3.0, 0.0, -50.0));
        let trees = views_p(&splats, &c2p, p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], limit);
        cut.deadline_stride = 1; // 讓注入的 deadline 閉包真的每一步都能咬到,而不是要等 64/4096 筆
        let mut ticks = 0;
        loop {
            // 兩個 deadline 閉包共用一個計數:用 Cell,否則兩個 &mut 閉包同時借 n 會 E0499
            let n = std::cell::Cell::new(0u32);
            let stop = || { n.set(n.get() + 1); n.get() > 1 };
            let st = cut.tick(&trees, &[0], 30, limit, 0.0, &mut || stop(), &mut || stop());
            ticks += 1;
            assert_valid(&cut, &parent);
            assert!(cut.cut_size <= 30);
            if st.settled { break; }
            assert!(ticks < 500, "不收斂");
        }
        assert!(ticks > 5, "每筆就停,應該要很多 tick:{ticks}");
        assert_eq!(cut_indices(&cut), atomic_cut(&splats, &p, limit, 30));
    }

    /// 遲滯:z=-100、limit 0.04 時 L1 的 ps ≈ 0.0399(貼著門檻)。先在 limit 0.03 建到 L2,
    /// 換 limit 0.04:eps 0.15 → 不收(0.0399 > 0.034);eps 0 → 收回 L1。
    #[test]
    fn inc_hysteresis_prevents_flip() {
        let (splats, _) = build_tree();
        let c2p = [0u32];
        let p = pose(Vec3A::new(0.0, 0.0, -100.0));
        let trees = views_p(&splats, &c2p, p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.03);
        settle(&mut cut, &trees, 1000, 0.03, 0.0);
        assert_eq!(cut_indices(&cut), (5..=20).collect::<Vec<_>>());
        for _ in 0..20 {
            let st = cut.tick(&trees, &[0], 1000, 0.04, 0.15, &mut || false, &mut || false);
            assert_eq!((st.expanded, st.collapsed), (0, 0));
        }
        assert_eq!(cut_indices(&cut), (5..=20).collect::<Vec<_>>(), "遲滯內不該動");
        settle(&mut cut, &trees, 1000, 0.04, 0.0);
        assert_eq!(cut_indices(&cut), vec![1, 2, 3, 4]);
    }

    /// D18 的上界是保守的:同一姿態鏈,開上界跳過與不開結果相同。
    /// 「不跳」的版本靠把每個 Node 的 radius 改成 f16::MAX 模擬 —— `dist − radius` 變負、被 `max(1e-6)`
    /// 夾住,ps_max 變成天文數字、`ps_max <= up` 永遠不成立 → 每組孩子都算。
    #[test]
    fn inc_hierarchical_bound_never_skips_needed_expansion() {
        let (splats, _) = build_tree();
        let c2p = [0u32];
        let limit = 0.03999;
        let chain = [Vec3A::new(0.0, 0.0, -100.0), Vec3A::new(0.0, 0.0, -50.0), Vec3A::new(5.0, 0.0, -100.0), Vec3A::new(0.0, 0.0, -150.0), Vec3A::new(0.0, 0.0, -100.0)];
        let mut with = IncrementalCut::new();
        let mut without = IncrementalCut::new();
        for (k, &o) in chain.iter().enumerate() {
            let trees = views_p(&splats, &c2p, pose(o));
            if k == 0 { with.restart(&trees, &[0], limit); without.restart(&trees, &[0], limit); }
            settle(&mut with, &trees, 1000, limit, 0.15);
            for n in without.nodes.iter_mut() { n.radius = f16::MAX; }
            settle(&mut without, &trees, 1000, limit, 0.15);
            assert_eq!(cut_indices(&with), cut_indices(&without), "站 {k}");
            let skipped = with.tick(&trees, &[0], 1000, limit, 0.15, &mut || false, &mut || false).scanned;
            assert!(skipped > 0);
        }
    }

    /// 穩了之後再 tick:有掃、沒拆收、settled、不需要 pack。
    #[test]
    fn inc_settled_stops() {
        let (splats, _) = build_tree();
        let c2p = [0u32];
        let p = pose(Vec3A::new(0.0, 0.0, -50.0));
        let trees = views_p(&splats, &c2p, p);
        let mut cut = IncrementalCut::new();
        cut.restart(&trees, &[0], 0.03);
        settle(&mut cut, &trees, 1000, 0.03, 0.0);
        cut.pack(&trees);
        let st = cut.tick(&trees, &[0], 1000, 0.03, 0.0, &mut || false, &mut || false);
        assert!(st.scanned > 0 && st.expanded == 0 && st.collapsed == 0 && st.settled && !st.changed);
        assert!(!cut.needs_pack());
    }
}

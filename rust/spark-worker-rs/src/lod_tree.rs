use std::{array, cell::{Ref, RefCell}, rc::Rc};

use ahash::AHashMap;
use half::f16;
use itertools::izip;
use js_sys::{Array, Object, Reflect, Uint32Array};
use wasm_bindgen::prelude::*;

pub(crate) use crate::lod_splat::LodSplat;
use crate::lod_traverse::{
    expand_until, is_fresh, seed_roots, InstanceParams, LoopExit, RoundMeta, TraverseCore, TreeView,
    MAX_INSTANCES, MAX_PAGED_INDEX,
};

const MAX_SPLAT_CHUNK: usize = 65536;

#[derive(Debug, Clone, Default)]
struct LodTree {
    splats: Rc<RefCell<Vec<LodSplat>>>,
    page_to_chunk: Vec<u32>,
    chunk_to_page: Vec<u32>,
}

struct LodState {
    next_id: u32,
    lod_trees: AHashMap<u32, LodTree>,
    /// traverse 的跨呼叫狀態(frontier/output/touched);round 存活期間不 clear。
    core: TraverseCore,
    /// 原子呼叫(`budget_ms <= 0`,如 raycast 那條)專用的第二顆 core。原子路徑**不讀不寫**
    /// `core` / `round`,所以旁路的原子 traverse 不會打斷切片中的 round(Task 5 review #1:
    /// `lodRaycast` 預設開、每 500ms 一次,若共用 core 會讓 > 500ms 的 round 永遠跑不完)。
    scratch: TraverseCore,
    /// 目前這一輪的固定參數;`None` = 沒有在跑的輪。
    round: Option<RoundMeta>,
    buffer: Vec<u32>,
}

impl LodState {
    fn new() -> Self {
        Self {
            next_id: 1000,
            lod_trees: AHashMap::new(),
            core: TraverseCore::default(),
            scratch: TraverseCore::default(),
            round: None,
            buffer: Vec::new(),
        }
    }
}

thread_local! {
    static STATE: RefCell<LodState> = RefCell::new(LodState::new());
}

fn set_lod_tree_data(state: &mut LodState, lod_id: u32, page_base: u32, _chunk_base: u32, count: u32, lod_tree_data: &Uint32Array) {
    let lod_tree = state.lod_trees.get(&lod_id).unwrap();
    let mut splats = lod_tree.splats.borrow_mut();

    if state.buffer.is_empty() {
        state.buffer.resize(MAX_SPLAT_CHUNK * 4, 0);
    }

    if page_base + count > splats.len() as u32 {
        let new_size = (splats.len() * 2).max((page_base + count) as usize);
        splats.resize_with(new_size, Default::default);
    }

    let mut index = 0;
    while index < count {
        let chunk = (count - index).min(MAX_SPLAT_CHUNK as u32);
        let buffer = &mut state.buffer[0..(chunk * 4) as usize];
        lod_tree_data.subarray((index * 4) as u32, ((index + chunk) * 4) as u32).copy_to(buffer);

        for i in 0..chunk {
            let i4 = i * 4;
            let words: [u32; 4] = array::from_fn(|j| buffer[i4 as usize + j]);
            let center = [
                f16::from_bits((words[0] & 0xffff) as u16),
                f16::from_bits((words[0] >> 16) as u16),
                f16::from_bits((words[1] & 0xffff) as u16),
            ];
            let size = f16::from_bits((words[1] >> 16) as u16);
            let child_count = (words[2] & 0xffff) as u16;
            let child_start = words[3];

            splats[(page_base + index + i) as usize] = LodSplat::new_f16(center, size, child_start, child_count);
        }
        index += chunk;
    }
}

#[wasm_bindgen]
pub fn new_lod_tree(capacity: u32) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let splats = Vec::with_capacity(capacity as usize);
        let splats = Rc::new(RefCell::new(splats));
        let page_capacity = capacity.div_ceil(65536);
        let page_to_chunk = Vec::with_capacity(page_capacity as usize);
        let chunk_to_page: Vec<u32> = Vec::with_capacity(page_capacity as usize);
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page });
        state.next_id += 1;

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

#[wasm_bindgen]
pub fn new_shared_lod_tree(orig_lod_id: u32) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_tree = state.lod_trees.get(&orig_lod_id).unwrap();
        let splats = lod_tree.splats.clone();
        let page_to_chunk = Vec::with_capacity(lod_tree.page_to_chunk.capacity());
        let chunk_to_page = Vec::with_capacity(lod_tree.chunk_to_page.capacity());

        let new_lod_id = state.next_id;
        state.next_id += 1;
        state.lod_trees.insert(new_lod_id, LodTree { splats, page_to_chunk, chunk_to_page });

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(new_lod_id)).unwrap();
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn init_lod_tree(num_splats: u32, lod_tree: Uint32Array) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let lod_id = state.next_id;
        let pages = num_splats.div_ceil(65536);
        let splats = Vec::with_capacity(num_splats as usize);
        let splats = Rc::new(RefCell::new(splats));
        let page_to_chunk = (0..pages).map(|page| page as u32).collect();
        let chunk_to_page: Vec<u32> = (0..pages).map(|chunk| chunk as u32).collect();
        state.lod_trees.insert(lod_id, LodTree { splats, page_to_chunk, chunk_to_page });
        state.next_id += 1;

        set_lod_tree_data(state, lod_id, 0, 0, num_splats, &lod_tree);

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_id)).unwrap();

        Ok(result)
    })
}

#[wasm_bindgen]
pub fn dispose_lod_tree(lod_id: u32) {
    STATE.with_borrow_mut(|state| {
        state.lod_trees.remove(&lod_id);
    })
}

#[wasm_bindgen]
pub fn update_lod_trees(lod_ids: &[u32], page_bases: &[u32], chunk_bases: &[u32], counts: &[u32], lod_trees: &Array) -> Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        for (&lod_id, &page_base, &chunk_base, &count, lod_tree_data) in izip!(lod_ids, page_bases, chunk_bases, counts, lod_trees.iter()) {
            let lod_tree = state.lod_trees.get_mut(&lod_id).unwrap();
            let pages = count.div_ceil(65536);
            let base_page = page_base >> 16;
            let base_chunk = chunk_base >> 16;

            if (base_page + pages) > lod_tree.page_to_chunk.len() as u32 {
                lod_tree.page_to_chunk.resize((base_page + pages) as usize, 0xFFFFFFFF);
            }
            if (base_chunk + pages) > lod_tree.chunk_to_page.len() as u32 {
                lod_tree.chunk_to_page.resize((base_chunk + pages) as usize, 0xFFFFFFFF);
            }

            if lod_tree_data.is_falsy() {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = 0xFFFFFFFF;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = 0xFFFFFFFF;
                }    
            } else {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = base_chunk + page;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = base_page + page;
                }

                let lod_tree_data = Uint32Array::from(lod_tree_data);
                set_lod_tree_data(state, lod_id, page_base, chunk_base, count, &lod_tree_data);
            }
        }

        let result = Object::new();
        // for (&lod_id, lod_tree) in state.lod_trees.iter() {
        //     let entry = Object::new();
        //     Reflect::set(&entry, &JsValue::from_str("pageToChunk"), &JsValue::from(lod_tree.page_to_chunk.clone())).unwrap();
        //     Reflect::set(&entry, &JsValue::from_str("chunkToPage"), &JsValue::from(lod_tree.chunk_to_page.clone())).unwrap();
        //     Reflect::set(&result, &JsValue::from_str(lod_id.to_string().as_str()), &JsValue::from(entry)).unwrap();
        // }
        Ok(result)
    })
}

#[wasm_bindgen]
pub fn get_lod_tree_level(lod_id: u32, level: u32) -> anyhow::Result<Object, JsValue> {
    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, .. } = state;
        let lod_tree = lod_trees.get(&lod_id).unwrap();
        let splats = lod_tree.splats.borrow();

        let root_size = splats[0].size();
        let level_size = root_size / (1.25f32.powi(level as i32));

        let mut nodes = vec![0];
        let mut output_nodes = Vec::new();

        while !nodes.is_empty() {
            let mut new_nodes = Vec::new();
            for node in nodes {
                let splat = &splats[node as usize];
                let &LodSplat { child_count, child_start, .. } = splat;
                if splat.size() <= level_size {
                    output_nodes.push(node);
                } else {
                    for child in child_start..child_start + child_count as u32 {
                        new_nodes.push(child);
                    }
                }
            }
            nodes = new_nodes;
        }

        let output = Uint32Array::new_with_length(output_nodes.len() as u32);
        for (i, node) in output_nodes.into_iter().enumerate() {
            output.set_index(i as u32, node);
        }

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
        Ok(result)
    })
}

/// 借出各 instance 的樹、拼成迴圈要的唯讀視圖。樹資料借 `lod_trees` **當下**的
/// (頁面更新可在兩片之間套用,spec §4.5);姿態用呼叫端給的 `params`(續跑時是 round 的)。
fn build_views<'a>(
    lod_trees: &'a AHashMap<u32, LodTree>, borrows: &'a [Ref<'a, Vec<LodSplat>>],
    lod_ids: &[u32], params: &[InstanceParams],
) -> Vec<TreeView<'a>> {
    lod_ids.iter().enumerate().map(|(i, id)| TreeView {
        lod_id: *id,
        splats: &borrows[i],
        chunk_to_page: &lod_trees.get(id).unwrap().chunk_to_page,
        params: params[i],
    }).collect()
}

/// 在 `core` 上跑一片:`fresh` 時先 reset + seed roots,再 best-first 展開到 `budget_ms`
/// 用完(`None` = 跑到底)。回 (迴圈怎麼結束, 這片花了幾 ms)。兩顆 core(切片 / 原子)共用。
#[allow(clippy::too_many_arguments)]
fn run_slice(
    core: &mut TraverseCore, lod_trees: &AHashMap<u32, LodTree>,
    lod_ids: &[u32], params: &[InstanceParams], root_pages: &[u32],
    max_splats: usize, pixel_scale_limit: f32, fresh: bool, budget_ms: Option<f32>,
) -> (LoopExit, f64) {
    let borrows: Vec<Ref<Vec<LodSplat>>> = lod_ids.iter()
        .map(|id| lod_trees.get(id).unwrap().splats.borrow())
        .collect();
    let trees = build_views(lod_trees, &borrows, lod_ids, params);

    if fresh {
        core.reset(max_splats);
        seed_roots(core, &trees, root_pages);
    }

    // 每 4096 次 pop 才查一次時間(spec D4):跨 wasm→JS 邊界的 Date::now() 不能每個節點都叫。
    let t0 = js_sys::Date::now();
    let deadline = budget_ms.map(|b| t0 + b as f64);
    let mut pops = 0u32;
    let exit = expand_until(core, &trees, max_splats, pixel_scale_limit, &mut || {
        pops = pops.wrapping_add(1);
        pops & 4095 == 0 && deadline.is_some_and(|d| js_sys::Date::now() >= d)
    });
    let slice_ms = js_sys::Date::now() - t0;
    drop(trees);
    drop(borrows);
    (exit, slice_ms)
}

/// 把 `core` 目前的快照(output ∪ frontier,不 drain,spec D3)與累計 touched 打包成 JS 物件。
fn pack_result(core: &TraverseCore, lod_ids: &[u32], done: bool, slice: u32, slice_ms: f64) -> Object {
    let num_instances = lod_ids.len();
    let output_size = core.output.len();
    let frontier_size = core.frontier.len();
    let cut = core.snapshot();

    let mut instance_counts = vec![0usize; num_instances];
    for &(inst_index, _) in cut.iter() {
        instance_counts[inst_index as usize] += 1;
    }
    let mut instance_outputs: Vec<Vec<u32>> = instance_counts.iter().map(|&n| Vec::with_capacity(n)).collect();
    for &(inst_index, paged_index) in cut.iter() {
        instance_outputs[inst_index as usize].push(paged_index);
    }

    let instance_indices = Array::new();
    for (inst_index, instance_output) in instance_outputs.iter().enumerate() {
        let rows = instance_output.len().div_ceil(16384);
        let capacity = rows * 16384;
        let output = Uint32Array::new_with_length(capacity as u32);
        output.subarray(0, instance_output.len() as u32).copy_from(instance_output);

        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("lodId"), &JsValue::from(lod_ids[inst_index])).unwrap();
        Reflect::set(&result, &JsValue::from_str("numSplats"), &JsValue::from(instance_output.len() as u32)).unwrap();
        Reflect::set(&result, &JsValue::from_str("indices"), &JsValue::from(output)).unwrap();
        instance_indices.push(&JsValue::from(result));
    }

    // chunks = 本輪**累計**的 touched(不是這片新增的)—— 這是 pager 不釋放 cut 所用頁的前提(spec §4.5)。
    let out_chunks = Array::new();
    for &(inst_index, chunk) in core.touched.iter() {
        let pair = Array::new();
        pair.push(&JsValue::from(inst_index));
        pair.push(&JsValue::from(chunk));
        out_chunks.push(&JsValue::from(pair));
    }

    let result = Object::new();
    Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(core.min_pixel_scale)).unwrap();
    Reflect::set(&result, &JsValue::from_str("instanceIndices"), &JsValue::from(instance_indices)).unwrap();
    Reflect::set(&result, &JsValue::from_str("chunks"), &JsValue::from(out_chunks)).unwrap();
    Reflect::set(&result, &JsValue::from_str("outputSize"), &JsValue::from(output_size)).unwrap();
    Reflect::set(&result, &JsValue::from_str("frontierSize"), &JsValue::from(frontier_size)).unwrap();
    Reflect::set(&result, &JsValue::from_str("leafCount"), &JsValue::from(core.leaf_count)).unwrap();
    Reflect::set(&result, &JsValue::from_str("done"), &JsValue::from(done)).unwrap();
    Reflect::set(&result, &JsValue::from_str("slice"), &JsValue::from(slice)).unwrap();
    Reflect::set(&result, &JsValue::from_str("sliceMs"), &JsValue::from(slice_ms)).unwrap();
    result
}

/// 可續跑的 traverse。`budget_ms <= 0` = 原子(在獨立的 `scratch` core 上跑到底,**不碰**
/// `core` / `round`,與 v2.1.0 逐位相同);`budget_ms > 0` = 切片,在 `core` 上按 `round` 續跑;
/// `restart` = 丟掉現有 round 從 root 重開。設計見本專案 spec §4。
#[wasm_bindgen]
pub fn traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
    budget_ms: f32, restart: bool,
) -> anyhow::Result<Object, JsValue> {
    let num_instances = lod_ids.len();
    if view_to_objects.len() != num_instances * 16 {
        return Err(JsValue::from_str("Invalid view_to_objects length"));
    }
    if lod_scales.len() != num_instances {
        return Err(JsValue::from_str("Invalid lod_scales length"));
    }
    if behind_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid behind_foveates length"));
    }
    if cone_foveates.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_foveates length"));
    }
    if cone_fov0s.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fov0s length"));
    }
    if cone_fovs.len() != num_instances {
        return Err(JsValue::from_str("Invalid cone_fovs length"));
    }
    // frontier heap 的 u64 打包位寬(lod_traverse::Entry):超過就明確拒絕,不靜默錯位。
    if num_instances > MAX_INSTANCES {
        return Err(JsValue::from_str(&format!("Too many LoD instances: {num_instances} > {MAX_INSTANCES}")));
    }

    // 這次呼叫自己的姿態 / foveation 參數(切片模式只在 round 開始時算一次、之後沿用 round 的)。
    let make_params = || -> Vec<InstanceParams> {
        (0..num_instances).map(|index| {
            let i16 = index * 16;
            let forward = glam::Vec3A::from_slice(&view_to_objects[(i16 + 8)..(i16 + 11)]).normalize().map(|x| -x);
            let origin = glam::Vec3A::from_slice(&view_to_objects[(i16 + 12)..(i16 + 15)]);
            let cone_dot0 = if cone_fov0s[index] > 0.0 { (0.5 * cone_fov0s[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            let cone_dot = if cone_fovs[index] > 0.0 { (0.5 * cone_fovs[index].clamp(0.0, 180.0)).to_radians().cos() } else { 1.0 };
            InstanceParams {
                origin,
                forward,
                lod_scale: lod_scales[index],
                behind_foveate: behind_foveates[index],
                cone_foveate: cone_foveates[index],
                cone_dot0,
                cone_dot: cone_dot.min(cone_dot0),
            }
        }).collect()
    };

    STATE.with_borrow_mut(|state| {
        let LodState { lod_trees, core, scratch, round, .. } = state;

        for id in lod_ids {
            let n = lod_trees.get(id).map_or(0, |t| t.splats.borrow().len());
            if n > MAX_PAGED_INDEX {
                return Err(JsValue::from_str(&format!(
                    "LoD tree {id} has {n} paged slots > {MAX_PAGED_INDEX} (512 pages); reduce pager maxSplats"
                )));
            }
        }

        if budget_ms <= 0.0 {
            // 原子模式:自己的參數、自己的 core、跑到底。`core` / `round` 原封不動,
            // 所以 raycast 那條旁路呼叫不會讓切片中的 round 失效(Task 5 review #1)。
            let params = make_params();
            let (exit, slice_ms) = run_slice(
                scratch, lod_trees, lod_ids, &params, root_pages,
                max_splats as usize, pixel_scale_limit, true, None,
            );
            debug_assert_eq!(exit, LoopExit::Done, "沒有 deadline 的 expand_until 只會以 Done 結束");
            return Ok(pack_result(scratch, lod_ids, true, 1, slice_ms));
        }

        // 切片模式。原子已在上面分流,所以 `is_fresh` 的 atomic 項固定給 false。
        let fresh = is_fresh(false, restart, round.as_ref(), lod_ids);
        if fresh {
            *round = Some(RoundMeta {
                lod_ids: lod_ids.to_vec(),
                params: make_params(),
                max_splats: max_splats as usize,
                pixel_scale_limit,
                slice: 0,
                done: false,
            });
        }
        let meta = round.as_mut().unwrap();

        // 視圖用 **round 的姿態**(續跑不吃當幀相機)。
        let (exit, slice_ms) = run_slice(
            core, lod_trees, &meta.lod_ids, &meta.params, root_pages,
            meta.max_splats, meta.pixel_scale_limit, fresh, Some(budget_ms),
        );
        meta.slice += 1;
        meta.done = exit == LoopExit::Done;

        Ok(pack_result(core, &meta.lod_ids, meta.done, meta.slice, slice_ms))
    })
}

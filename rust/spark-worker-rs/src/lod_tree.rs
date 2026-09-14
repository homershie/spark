use std::{array, cell::{Ref, RefCell}, rc::Rc};

use ahash::AHashMap;
use half::f16;
use itertools::izip;
use js_sys::{Array, Object, Reflect, Uint32Array};
use wasm_bindgen::prelude::*;

pub(crate) use crate::lod_splat::LodSplat;
use crate::lod_cut::IncrementalCut;
use crate::lod_traverse::{
    expand_until, seed_roots, InstanceParams, LoopExit, TraverseCore, TreeView,
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
    /// 原子路徑(`budget_ms <= 0` / `incremental=false`,含 raycast 那條旁路)專用的 core,
    /// 每次呼叫 reset + 跑到底。原子路徑**不讀不寫** `cut`,所以旁路的原子 traverse 不會
    /// 打斷正在累積的增量 cut(Task 5 review #1 的道理延續到方案 B)。
    scratch: TraverseCore,
    /// 方案 B:跨呼叫保留的增量 cut(`incremental=true` 的路徑)。
    cut: IncrementalCut,
    buffer: Vec<u32>,
}

impl LodState {
    fn new() -> Self {
        Self {
            next_id: 1000,
            lod_trees: AHashMap::new(),
            scratch: TraverseCore::default(),
            cut: IncrementalCut::new(),
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
        // 頁釋放(`is_falsy()` 分支)的 (lod_id, chunk) 收集在這裡,等這個迴圈(以及裡面每次
        // 借用的 `lod_tree`)結束之後才餵給 `state.cut`——`note_chunk_released` 需要 `&mut state.cut`,
        // 跟迴圈裡借著的 `state.lod_trees` 分開處理,不用在借用還活著時硬擠進同一行。
        let mut released: Vec<(u32, u32)> = Vec::new();
        // 對稱地收集頁「補回」的 (lod_id, chunk)(`lod_tree_data` 為真的分支,每頁寫入資料
        // 就算一次到達)——`note_chunk_resident` 同樣需要 `&mut state.cut`,理由同上(review
        // 最終輪 F1 + F2)。
        let mut resident: Vec<(u32, u32)> = Vec::new();

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
                    released.push((lod_id, base_chunk + page));
                }
            } else {
                for page in 0..pages {
                    lod_tree.page_to_chunk[(base_page + page) as usize] = base_chunk + page;
                    lod_tree.chunk_to_page[(base_chunk + page) as usize] = base_page + page;
                    resident.push((lod_id, base_chunk + page));
                }

                let lod_tree_data = Uint32Array::from(lod_tree_data);
                set_lod_tree_data(state, lod_id, page_base, chunk_base, count, &lod_tree_data);
            }
        }

        // 方案 B:被釋放的頁若是增量 cut 正在引用的 chunk,強制下次 `traverse_lod_trees` 重 pack
        // (Task 4 review round 1 F1)——否則 `generation` 沒變、`needs_pack()` 誤判沒事,JS 繼續
        // 渲染指向舊 chunk 的 paged index,而那個槽位現在住著別的資料。
        for (lod_id, chunk) in released {
            state.cut.note_chunk_released(lod_id, chunk);
        }
        // review 最終輪 F1 + F2:對稱地通知頁「補回」——cut 原本只從頁釋放得知變化,從沒被告知
        // 頁到達,見 `IncrementalCut::note_chunk_resident` 的文件註解。
        for (lod_id, chunk) in resident {
            state.cut.note_chunk_resident(lod_id, chunk);
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

/// 原子 traverse:在 `scratch` core 上 reset + seed roots + best-first 展到 Done(無 deadline)。
/// `budget_ms ≤ 0` 或 `incremental=false`(含 raycast 那條旁路)都走這裡,與 v2.1.0 逐位相同。
/// `trees` 由呼叫端建好(借用 `lod_trees` 當下的資料),與方案 B 的 `cut.tick()` 共用同一份,
/// 不必為了原子路徑再借一次。
fn run_atomic(scratch: &mut TraverseCore, trees: &[TreeView], root_pages: &[u32], max_splats: usize, pixel_scale_limit: f32) {
    scratch.reset(max_splats);
    seed_roots(scratch, trees, root_pages);
    let exit = expand_until(scratch, trees, max_splats, pixel_scale_limit, &mut || false);
    debug_assert_eq!(exit, LoopExit::Done, "沒有 deadline 的 expand_until 只會以 Done 結束");
}

/// 把每 instance 的 paged index 補到 16384 倍數、包成 `{lodId, numSplats, indices}` 的 `Array`。
/// 原子路徑把 `TraverseCore::snapshot()` 攤成 `Vec<Vec<u32>>` 再呼叫;增量路徑直接餵
/// `IncrementalCut::pack()` 的 `Packed::indices`——兩邊都是「每 instance 一份 paged index」。
fn pack_indices(instance_outputs: Vec<Vec<u32>>, lod_ids: &[u32]) -> Array {
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
    instance_indices
}

/// `(lod_id, chunk)` pairs → JS `Array` of `[lodId, chunk]` pairs。原子路徑餵 `scratch.touched`
/// (本輪**累計**的 touched,不是這片新增的——這是 pager 不釋放 cut 所用頁的前提,spec §4.5);
/// 增量路徑餵 `IncrementalCut::chunks()` 的 `Chunks::list`(roots / needed / wanted 三段)。
fn pack_chunks(list: &[(u32, u32)]) -> Array {
    let out_chunks = Array::new();
    for &(lod_id, chunk) in list {
        let pair = Array::new();
        pair.push(&JsValue::from(lod_id));
        pair.push(&JsValue::from(chunk));
        out_chunks.push(&JsValue::from(pair));
    }
    out_chunks
}

/// 原子(`budget_ms ≤ 0` 或 `incremental=false`)/ 增量(`IncrementalCut::tick`,方案 B)分流。
/// 切片 round 路徑(`RoundMeta` / `is_fresh` / `run_slice`)已刪除——增量路徑要不要從 root
/// 重開,由 `IncrementalCut::tick`(內部呼叫 `restart`)依 instance 集合是否變了自己判斷,
/// 呼叫端不用再傳 `restart` 旗標。設計見本專案 spec §4、task brief(方案 B Task 4)。
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn traverse_lod_trees(
    max_splats: u32, pixel_scale_limit: f32, _last_pixel_limit: Option<f32>,
    lod_ids: &[u32], root_pages: &[u32],
    view_to_objects: &[f32], lod_scales: &[f32],
    behind_foveates: &[f32], cone_foveates: &[f32],
    cone_fov0s: &[f32], cone_fovs: &[f32],
    budget_ms: f32, incremental: bool, hysteresis: f32,
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

    // 這次呼叫自己的姿態 / foveation 參數(每次呼叫都重算——相機每幀在動;增量路徑姿態變了
    // 與否由 `IncrementalCut::tick` 自己比對 `self.params`,呼叫端不用記上一次的)。
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
        let LodState { lod_trees, scratch, cut, .. } = state;

        for id in lod_ids {
            let n = lod_trees.get(id).map_or(0, |t| t.splats.borrow().len());
            if n > MAX_PAGED_INDEX {
                return Err(JsValue::from_str(&format!(
                    "LoD tree {id} has {n} paged slots > {MAX_PAGED_INDEX} (512 pages); reduce pager maxSplats"
                )));
            }
        }

        let params = make_params();
        let borrows: Vec<Ref<Vec<LodSplat>>> = lod_ids.iter().map(|id| lod_trees.get(id).unwrap().splats.borrow()).collect();
        let trees = build_views(lod_trees, &borrows, lod_ids, &params);
        let t0 = js_sys::Date::now();

        if budget_ms <= 0.0 || !incremental {
            // 原子模式:獨立的 `scratch` core、每次重開跑到底。與 v2.1.0 逐位相同,也不碰
            // `cut`,所以旁路的原子呼叫(raycast)不會弄壞正在累積的增量 cut(Task 5 review #1
            // 的道理延續到方案 B)。
            run_atomic(scratch, &trees, root_pages, max_splats as usize, pixel_scale_limit);
            let cut_vec = scratch.snapshot();
            let mut outs: Vec<Vec<u32>> = lod_ids.iter().map(|_| Vec::new()).collect();
            for &(inst, paged) in &cut_vec {
                outs[inst as usize].push(paged);
            }
            let result = Object::new();
            Reflect::set(&result, &JsValue::from_str("instanceIndices"), &pack_indices(outs, lod_ids)).unwrap();
            Reflect::set(&result, &JsValue::from_str("chunks"), &pack_chunks(&scratch.touched)).unwrap();
            Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(scratch.min_pixel_scale)).unwrap();
            Reflect::set(&result, &JsValue::from_str("neededChunks"), &JsValue::from(scratch.touched.len() as u32)).unwrap();
            Reflect::set(&result, &JsValue::from_str("done"), &JsValue::from(true)).unwrap();
            Reflect::set(&result, &JsValue::from_str("tickMs"), &JsValue::from(js_sys::Date::now() - t0)).unwrap();
            return Ok(result);
        }

        // 增量模式(方案 B):掃描只拿 60% 預算(否則大 cut 的掃描把時間吃光、拆收永遠輪不到,
        // spec §4.2);其餘階段(收/拆/門檻控制器)吃剩下的到 100%。
        let scan_deadline = t0 + 0.6 * budget_ms as f64;
        let deadline = t0 + budget_ms as f64;
        let stats = cut.tick(
            &trees, root_pages, max_splats as usize, pixel_scale_limit, hysteresis,
            &mut || js_sys::Date::now() >= scan_deadline, &mut || js_sys::Date::now() >= deadline,
        );
        // pack 之後才知道 evicted(頁被踢的洞);`TickStats::evicted` 是給 `tick()` 自己用的
        // 佔位欄位(它本身不 pack),這裡算出來的才是真值,放進下面的 `tick` 物件。
        let (indices, evicted) = if cut.needs_pack() {
            let packed = cut.pack(&trees);
            (JsValue::from(pack_indices(packed.indices, lod_ids)), packed.evicted)
        } else {
            (JsValue::NULL, 0)
        };
        let chunks = cut.chunks();
        let result = Object::new();
        Reflect::set(&result, &JsValue::from_str("instanceIndices"), &indices).unwrap();
        Reflect::set(&result, &JsValue::from_str("chunks"), &pack_chunks(&chunks.list)).unwrap();
        Reflect::set(&result, &JsValue::from_str("pixelLimit"), &JsValue::from(stats.t)).unwrap();
        Reflect::set(&result, &JsValue::from_str("neededChunks"), &JsValue::from((chunks.roots + chunks.needed) as u32)).unwrap();
        Reflect::set(&result, &JsValue::from_str("done"), &JsValue::from(stats.settled)).unwrap();

        let tick = Object::new();
        for (k, v) in [
            ("scanned", stats.scanned as f64), ("expanded", stats.expanded as f64), ("collapsed", stats.collapsed as f64),
            ("passes", stats.passes as f64), ("evicted", evicted as f64), ("wantedCount", stats.wanted_count as f64),
            ("cutSize", stats.cut_size as f64), ("t", stats.t as f64), ("arenaLeaked", stats.arena_leaked as f64),
            ("boundSkipped", stats.bound_skipped as f64), ("tickMs", js_sys::Date::now() - t0),
        ] {
            Reflect::set(&tick, &JsValue::from_str(k), &JsValue::from(v)).unwrap();
        }
        Reflect::set(&tick, &JsValue::from_str("settled"), &JsValue::from(stats.settled)).unwrap();
        Reflect::set(&tick, &JsValue::from_str("changed"), &JsValue::from(stats.changed)).unwrap();
        Reflect::set(&result, &JsValue::from_str("tick"), &tick).unwrap();
        drop(trees);
        drop(borrows);
        Ok(result)
    })
}

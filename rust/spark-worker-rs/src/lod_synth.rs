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
    compute_pixel_scale, expand_until, limit_key, seed_roots, InstanceParams, TraverseCore, TreeView,
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

    // ── 方案 B 的成本模型:對現成的 2.5M cut 用新姿態重算 pixel_scale 要多久 ──
    // (a) cut 只存 paged index,重算時回 `splats[paged]` 讀節點(隨機存取 268MB 陣列)
    // (b) cut 表自帶節點的 center/size(16B 一筆),順序掃
    let p0 = params(POSES[0].0, POSES[0].1);
    let (cut, _) = run_atomic(&splats, &c2p, root_page, p0, 2_500_000);
    // 用 traverse 產出的順序(兄弟連續),不是排序後的 —— 排序會美化 (a)
    let trees = [TreeView { lod_id: 1, splats: &splats, chunk_to_page: &c2p, params: p0 }];
    let mut core = TraverseCore::default();
    core.reset(2_500_000);
    seed_roots(&mut core, &trees, &[root_page]);
    expand_until(&mut core, &trees, 2_500_000, limit_key(PIXEL_SCALE_LIMIT), &mut || false);
    let order: Vec<u32> = core.snapshot().into_iter().map(|(_, paged)| paged).collect();
    assert_eq!(order.len(), cut.len());
    let copied: Vec<LodSplat> = order.iter().map(|&paged| splats[paged as usize].clone()).collect();
    let p1 = params(POSES[1].0, POSES[1].1);
    let mut best_a = f64::INFINITY;
    let mut best_b = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..rounds {
        let t = Instant::now();
        for &paged in &order {
            sink += compute_pixel_scale(&splats[paged as usize], &p1);
        }
        best_a = best_a.min(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        for node in &copied {
            sink += compute_pixel_scale(node, &p1);
        }
        best_b = best_b.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!(
        "RECOMPUTE {} nodes: (a) via splats[paged] {best_a:.1} ms  (b) sequential copy {best_b:.1} ms  (sink {sink:e})",
        order.len()
    );
}

/// 方案 B 的走路模擬:20 站、每站位移 0.05 單位 + 轉 1°,每站 tick(20ms 預算)到 settled。
/// 印每 tick 的 ms(tick / pack 分開)、每站 settle 的 tick 數、拆收數。目標:tick ≤ 30ms、settle ≤ 6 tick。
/// worker crate 的 test build 不呼叫它(只有 `traverse-bench` 用 `#[path]` include 後在 `main` 呼叫)。
///
/// ⚠️ **station 0 是冷啟動,不是缺陷**(Task 5 review round 1):空 cut 直接跳到第一個姿態,
/// 工作量等同一次完整的原子 traverse(2.5M 預算全部要從無到有拆出來),需要 50+ tick 是預期
/// 行為。給它自己的上限(400 tick,一般站的 60 遠遠不夠它收斂)與獨立計數
/// (`COLD START: N ticks`),**不**計進 `settle ticks` 的 median/max、也**不**計進
/// `all_tick`/`all_pack`(這兩份統計代表「走路中途轉頭一次」的穩態成本,冷啟動混進去會把
/// p90 拉到失真)。
///
/// `eps` = 遲滯帶(`cut.tick` 的 hysteresis 參數,production/預設 0.05,spec D6 2026-09-14
/// 更新——0.15 只留 85% 與原子的重疊,0.05 有 95–98%);`diag` 開啟時每站結束多印一行
/// `t`/`cut_size`/該站姿態下的原子 cut 大小(Task 5 review round 2 的診斷,見
/// `docs/superpowers/specs/2026-09-11-incremental-traverse-design.md` §7.2)。
///
/// ⚠️ **Jaccard ≥ 0.93 這道 `assert!` 只在 `eps == 0.0` 才是「跟原子等價」的判準**
/// (Task 5 review round 2 的裁決,門檻值由 0.97 下修為 0.93):`eps > 0` 時遲滯帶內的
/// 節點依定義就是不動的,增量 cut 跟零遲滯的原子參考本來就會有結構性差距,不是缺陷,
/// 只印 `(eps>0, informational)`,不斷言。**`eps == 0` 時殘留的 3–5% membership 差距
/// 也不是缺陷**,是 spec §4.5 已經記錄的固有分岔:原子是「第一個裝不下就停」(greedy,
/// 貪婪展到 budget 用完的那一刻,不看有沒有更接近均衡),增量是「`t` = misfit 的
/// ps」(用上一次裝不下的候選 ps 當門檻,逼近但不等於同一個均衡點)——兩條路徑在
/// **門檻帶上的邊界節點**(最遠即將被收的、最小即將被拆的那一小撮)取捨不同,count
/// 幾乎相同、差的正是這批邊界節點。`eps = 0` 本身也是退化設定(遲滯帶寬度 0),
/// 可能讓貼著門檻的節點在展開/收回之間平手乒乓、不容易乾淨 `settled`(故 settle 上限
/// 給到 200)。**不論 eps 為何都斷言** `cut.cut_size >= 0.95·max`(spec 的品質下限,
/// 見 §7.1 `synth_inc_walk`)——這條與遲滯帶/門檻分岔無關,純粹檢查有沒有把預算用滿。
#[allow(dead_code)]
pub(crate) fn walk_main(rounds: usize, eps: f32, diag: bool) {
    use crate::lod_cut::IncrementalCut;
    use std::time::Instant;
    const COLD_START_TICK_CAP: u32 = 400;
    // wasm 在 eps=0 需要 > 60 tick 才收斂(Task 5 review round 1/2 實測);200 給足餘裕。
    const NORMAL_TICK_CAP: u32 = 200;
    let tree = build_synth(7, 256 * CHUNK, 256.0);
    let (splats, c2p, root_page) = page_out(&tree);
    let mut cut = IncrementalCut::new();
    let limit = PIXEL_SCALE_LIMIT;
    let max = 2_500_000;
    let mut origin = POSES[0].0;
    let mut yaw = 0.0f32;
    let mut all_tick = Vec::new();
    let mut all_pack = Vec::new();
    let mut settle_ticks = Vec::new();
    let mut cold_start_tick: Vec<f64> = Vec::new();
    // 最後一站實際用來 tick 的姿態(推進之前存起來,見 task brief 的提醒:
    // 推進之後的 origin/yaw 不是最後一站 tick 時用的那個,Jaccard 會量錯東西)。
    let mut last = (origin, yaw);
    eprintln!("WALK PARAMS: eps={eps} diag={diag}");
    for station in 0..(20 * rounds) {
        let cold_start = station == 0;
        let tick_cap = if cold_start { COLD_START_TICK_CAP } else { NORMAL_TICK_CAP };
        let forward = Vec3A::new(yaw.sin(), 0.0, yaw.cos());
        let trees = [TreeView { lod_id: 1, splats: &splats, chunk_to_page: &c2p, params: params(origin, forward) }];
        let mut ticks = 0;
        loop {
            let t = Instant::now();
            let scan_at = t + std::time::Duration::from_millis(12);
            let deadline_at = t + std::time::Duration::from_millis(20);
            let st = cut.tick(&trees, &[root_page], max, limit, eps, &mut || Instant::now() >= scan_at, &mut || Instant::now() >= deadline_at);
            let tick_ms = t.elapsed().as_secs_f64() * 1e3;
            let t2 = Instant::now();
            let packed = if cut.needs_pack() { Some(cut.pack(&trees)) } else { None };
            let pack_ms = t2.elapsed().as_secs_f64() * 1e3;
            ticks += 1;
            if cold_start {
                cold_start_tick.push(tick_ms);
            } else {
                all_tick.push(tick_ms);
                if packed.is_some() { all_pack.push(pack_ms); }
            }
            if station < 2 || ticks <= 2 {
                eprintln!("station {station} tick {ticks}: tick {tick_ms:6.1} ms pack {pack_ms:5.1} ms  cut {}  exp {} col {} t {:.3e} settled {}",
                    st.cut_size, st.expanded, st.collapsed, st.t, st.settled);
            }
            if st.settled || ticks >= tick_cap { break; }
        }
        if cold_start {
            let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
            eprintln!("COLD START: {ticks} ticks (tick median {:.1} ms, excluded from WALK: stats below)",
                med(&mut cold_start_tick.clone()));
        } else {
            settle_ticks.push(ticks);
        }
        if diag {
            // 該站姿態下的原子 cut 大小(不看去重/遲滯,純粹「這個預算 + 這個門檻能撐到的最終
            // 節點數」)—— 跟 `cut.t`/`cut.cut_size` 對照,能看出 `t` 有沒有卡在真正均衡點之上
            // (decay gate `cut < 0.98·max` 在 cut 落在 [0.98·max, max] 之間會擋住衰減)。
            let (_, atomic_n) = run_atomic(&splats, &c2p, root_page, trees[0].params, max);
            eprintln!("station {station} DIAG: t={:.6} cut_size={} atomic_cut={atomic_n}", cut.t, cut.cut_size);
        }
        last = (origin, yaw);
        origin += Vec3A::new(0.05 * yaw.cos(), 0.0, -0.05 * yaw.sin());
        yaw += 1.0f32.to_radians();
    }
    let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let p90 = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[(v.len() as f64 * 0.9) as usize] };
    settle_ticks.sort_unstable();
    eprintln!("WALK: tick median {:.1} p90 {:.1} ms | pack median {:.1} ms | settle ticks median {} max {} | cut {}",
        med(&mut all_tick.clone()), p90(&mut all_tick.clone()), med(&mut all_pack.clone()),
        settle_ticks[settle_ticks.len() / 2], settle_ticks.last().unwrap(), cut.cut_size);
    // 品質下限(spec §7.1 synth_inc_walk):最後一站的增量 cut 與原子 cut 的 Jaccard ≥ 0.93(eps=0)
    let (last_origin, last_yaw) = last;
    let forward = Vec3A::new(last_yaw.sin(), 0.0, last_yaw.cos());
    let p_last = params(last_origin, forward);
    let trees = [TreeView { lod_id: 1, splats: &splats, chunk_to_page: &c2p, params: p_last }];
    let (atomic, _) = run_atomic(&splats, &c2p, root_page, p_last, max);
    let inc: AHashSet<u32> = cut.pack(&trees).indices[0].iter().copied().collect();
    let at: AHashSet<u32> = atomic.into_iter().collect();
    let inter = inc.intersection(&at).count() as f64;
    let j = inter / (inc.len() as f64 + at.len() as f64 - inter);
    // Jaccard 只在 eps == 0.0 才是「跟原子等價」的判準(遲滯帶內的節點依定義不動,
    // eps > 0 時本來就會跟零遲滯的原子參考有結構性差距,不是缺陷)——Task 5 review
    // round 2 裁決:eps > 0 只印、不斷言。
    //
    // eps == 0 時門檻是 0.93,不是 1.0(甚至不是原本的 0.97):原子「第一個裝不下就
    // 停」(greedy,貪婪展到 budget 用完那一刻,不管有沒有更接近均衡)vs 增量「t =
    // misfit 的 ps」(用上一次裝不下的候選 ps 當門檻,逼近但不等於同一個均衡點)——
    // 這是 spec §4.5 已經記錄的固有分岔,兩條路徑在門檻帶上的邊界節點(最遠即將被
    // 收的、最小即將被拆的那一小撮)取捨不同,count 幾乎相同、差的正是這批邊界節點,
    // 不是缺陷。`eps = 0` 本身也是退化設定,貼著門檻的節點可能在展開/收回之間平手
    // 乒乓、不容易乾淨 settled(這也是 settle 上限給到 200 的理由)。
    if eps == 0.0 {
        eprintln!("JACCARD vs atomic: {j:.3}  (inc {} atomic {})", inc.len(), at.len());
        assert!(j >= 0.93, "增量 cut 與原子差太多:{j:.3}");
    } else {
        eprintln!("JACCARD vs atomic: {j:.3}  (inc {} atomic {})  (eps>0, informational)", inc.len(), at.len());
    }
    // 品質下限(spec §7.1 synth_inc_walk):不論 eps 為何,最後一站的 cut 都不該明顯小於
    // 預算——這條跟遲滯帶無關,遲滯只影響「跟原子精確重疊多少」,不影響「有沒有把預算用滿」。
    assert!(
        cut.cut_size >= (0.95 * max as f64) as usize,
        "cut_size {} 明顯小於預算(0.95×max = {}),沒用滿預算",
        cut.cut_size, (0.95 * max as f64) as usize
    );
}

/// 「填滿」模擬(D22,2026-09-14 手測 `lodSplatScaleWalk` 4 的重現):同一個姿態(POSES[0])
/// 先在 2.5M 預算 settle,再把 `max_splats` 直接拉到 10M、姿態不動,數 tick(20ms 預算、真
/// deadline:掃描 12ms / 全部 20ms,跟 `walk_main` 同一組)直到 `cut_size ≥ 0.98·目標`(目標 =
/// min(10M, 該姿態原子 10M 的 cut 大小 —— 若門檻先到、原子也填不到 10M))或 settled。
/// 印 tick 數與 wasm 總毫秒;同一輪先量原子 10M 一次到底要幾毫秒,當作比較基準(目標:增量
/// 填滿 ≤ 2× 原子)。D22 之前:每 tick 掃描先吃 60% 預算才輪到拆、每個有變動的 tick 都白 pack
/// 整份 10M 索引(JS 每 50ms 才上傳一次),填滿要分鐘級;之後:有積壓就跳過掃描、pack 只在
/// 呼叫端要上傳時做。`pack_every` 模擬 JS 的 `lodApplyIntervalMs`:每隔幾個 tick pack 一次
/// (0 = 每 tick pack,D22 之前的行為)。
#[allow(dead_code)]
// `u32::is_multiple_of` 是 1.87 才穩定,workspace 的 rust-version 是 1.82(同 lod_cut.rs 的兩處)。
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn fill_main(rounds: usize, eps: f32, pack_every: usize) {
    use crate::lod_cut::IncrementalCut;
    use std::time::Instant;
    const SETTLE_TICK_CAP: u32 = 400;
    const FILL_TICK_CAP: u32 = 20_000;
    let base = 2_500_000usize;
    let target_max = 10_000_000usize;
    let tree = build_synth(7, 256 * CHUNK, 256.0);
    let (splats, c2p, root_page) = page_out(&tree);
    let limit = PIXEL_SCALE_LIMIT;
    let p = params(POSES[0].0, POSES[0].1);
    let trees = [TreeView { lod_id: 1, splats: &splats, chunk_to_page: &c2p, params: p }];
    eprintln!("FILL PARAMS: eps={eps} pack_every={pack_every} base={base} target_max={target_max}");
    let mut best_atomic = f64::INFINITY;
    let mut atomic_n = 0usize;
    let mut best_fill: Option<(u32, f64, f64, usize)> = None;
    for round in 0..rounds {
        let t = Instant::now();
        let (_, n10) = run_atomic(&splats, &c2p, root_page, p, target_max);
        let atomic_ms = t.elapsed().as_secs_f64() * 1e3;
        best_atomic = best_atomic.min(atomic_ms);
        atomic_n = n10;
        let goal = (0.98 * target_max.min(n10) as f64) as usize;

        let mut cut = IncrementalCut::new();
        let tick_once = |cut: &mut IncrementalCut, max: usize, k: u32| -> (crate::lod_cut::TickStats, f64, f64) {
            let t = Instant::now();
            let scan_at = t + std::time::Duration::from_millis(12);
            let deadline_at = t + std::time::Duration::from_millis(20);
            let st = cut.tick(&trees, &[root_page], max, limit, eps, &mut || Instant::now() >= scan_at, &mut || Instant::now() >= deadline_at);
            let tick_ms = t.elapsed().as_secs_f64() * 1e3;
            let t2 = Instant::now();
            let want_pack = pack_every == 0 || k % pack_every as u32 == 0;
            if want_pack && cut.needs_pack() {
                let _ = cut.pack(&trees);
            }
            (st, tick_ms, t2.elapsed().as_secs_f64() * 1e3)
        };
        // 1. 2.5M settle(冷啟動,不計)
        let mut k = 0u32;
        loop {
            k += 1;
            let (st, _, _) = tick_once(&mut cut, base, k);
            if st.settled || k >= SETTLE_TICK_CAP { break; }
        }
        let settled_n = cut.cut_size;
        // 2. 同姿態、max → 10M,數到 0.98·goal 或 settled
        let mut ticks = 0u32;
        let mut sum_tick = 0.0;
        let mut sum_pack = 0.0;
        let wall = Instant::now();
        let reason;
        loop {
            ticks += 1;
            let (st, tick_ms, pack_ms) = tick_once(&mut cut, target_max, ticks);
            sum_tick += tick_ms;
            sum_pack += pack_ms;
            if ticks <= 3 || ticks % 50 == 0 {
                eprintln!("fill tick {ticks}: tick {tick_ms:6.1} ms pack {pack_ms:5.1} ms  cut {}  scan {} exp {} col {} t {:.3e} settled {}",
                    st.cut_size, st.scanned, st.expanded, st.collapsed, st.t, st.settled);
            }
            if cut.cut_size >= goal { reason = "reached"; break; }
            if st.settled { reason = "settled"; break; }
            if ticks >= FILL_TICK_CAP { reason = "cap"; break; }
        }
        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "round {round}: atomic10M {atomic_ms:.1} ms (cut {n10}) | settled@2.5M cut {settled_n} in {k} ticks | FILL {ticks} ticks ({reason}) tick-sum {sum_tick:.1} ms pack-sum {sum_pack:.1} ms wall {wall_ms:.1} ms cut {} goal {goal}",
            cut.cut_size
        );
        assert!(cut.cut_size >= goal, "填不到 0.98×目標:cut {} goal {goal}({reason})", cut.cut_size);
        if best_fill.is_none_or(|b| sum_tick + sum_pack < b.1 + b.2) {
            best_fill = Some((ticks, sum_tick, sum_pack, cut.cut_size));
        }
    }
    let (ticks, sum_tick, sum_pack, n) = best_fill.unwrap();
    let total = sum_tick + sum_pack;
    eprintln!(
        "FILL MIN of {rounds}: {ticks} ticks, tick {sum_tick:.1} + pack {sum_pack:.1} = {total:.1} ms (cut {n}) | atomic 10M {best_atomic:.1} ms (cut {atomic_n}) | ratio {:.2}x",
        total / best_atomic
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

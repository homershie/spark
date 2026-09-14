//! LoD traverse 基準(原生 / wasm32-wasip1 同一份)。見 Cargo.toml 與 run.sh。
#![allow(dead_code)]

#[path = "../../spark-worker-rs/src/lod_splat.rs"]
mod lod_splat;
#[path = "../../spark-worker-rs/src/lod_traverse.rs"]
mod lod_traverse;
#[path = "../../spark-worker-rs/src/lod_cut.rs"]
mod lod_cut;
#[path = "../../spark-worker-rs/src/lod_synth.rs"]
mod lod_synth;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("walk") {
        let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
        // args[3] = eps(遲滯帶,預設 0.05 = production 值,spec D6 2026-09-14 更新);
        // args[4] = "1" 開 per-station t/cut_size/atomic 診斷(Task 5 review round 2,
        // 見 lod_synth.rs::walk_main 文件)。
        let eps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.05);
        let diag = args.get(4).map(String::as_str) == Some("1");
        lod_synth::walk_main(rounds, eps, diag);
    } else if args.get(1).map(String::as_str) == Some("stream") {
        // stream 模式(D23):串流冷啟動——只有 chunk 0 resident,每 tick 依 wanted 順序送到 per_tick 個
        // chunk(模擬 3 個 fetcher 一 tick 後到達),數填到 0.95·2.5M 的 tick / ms。
        // args[2] = rounds(預設 1);args[3] = eps(預設 0.05);args[4] = per_tick(預設 3)。
        let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
        let eps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.05);
        let per_tick = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
        lod_synth::stream_main(rounds, eps, per_tick);
    } else if args.get(1).map(String::as_str) == Some("fill") {
        // fill 模式(D22):同姿態 2.5M settle → max 10M,數填滿的 tick 與毫秒,對照原子 10M。
        // args[2] = rounds(預設 1);args[3] = eps(預設 0.05);args[4] = pack_every
        // (每幾個 tick pack 一次,預設 3 ≈ JS 的 lodApplyIntervalMs 50ms / tick 20ms;0 = 每 tick pack)。
        let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
        let eps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.05);
        let pack_every = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
        lod_synth::fill_main(rounds, eps, pack_every);
    } else {
        let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
        lod_synth::bench_main(rounds);
    }
}

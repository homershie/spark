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
        // args[3] = eps(遲滯帶,預設 0.15 = production 值);args[4] = "1" 開 per-station
        // t/cut_size/atomic 診斷(Task 5 review round 2,見 lod_synth.rs::walk_main 文件)。
        let eps = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.15);
        let diag = args.get(4).map(String::as_str) == Some("1");
        lod_synth::walk_main(rounds, eps, diag);
    } else {
        let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
        lod_synth::bench_main(rounds);
    }
}

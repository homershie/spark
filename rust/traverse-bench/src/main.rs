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
        lod_synth::walk_main(rounds);
    } else {
        let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
        lod_synth::bench_main(rounds);
    }
}

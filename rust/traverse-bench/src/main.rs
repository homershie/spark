//! LoD traverse 基準(原生 / wasm32-wasip1 同一份)。見 Cargo.toml 與 run.sh。
#![allow(dead_code)]

#[path = "../../spark-worker-rs/src/lod_splat.rs"]
mod lod_splat;
#[path = "../../spark-worker-rs/src/lod_traverse.rs"]
mod lod_traverse;
#[path = "../../spark-worker-rs/src/lod_synth.rs"]
mod lod_synth;

fn main() {
    let rounds = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    lod_synth::bench_main(rounds);
}

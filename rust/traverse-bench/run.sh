#!/usr/bin/env bash
# LoD traverse 基準:原生 + wasm32-wasip1(Node 內建 WASI 跑,不用裝 wasmtime)。
# 用法:rust/traverse-bench/run.sh [native|wasm|both] [rounds]
# wasm 用與 build_rust_wasm.sh 相同的 target-feature(simd128 + bulk-memory),數字才跟瀏覽器裡的
# worker 同一種碼;Node 的 V8 與瀏覽器同一顆引擎。
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-both}"
rounds="${2:-5}"
if [[ "$mode" == "native" || "$mode" == "both" ]]; then
  echo "== native =="
  cargo run --release -q -p traverse-bench -- "$rounds"
fi
if [[ "$mode" == "wasm" || "$mode" == "both" ]]; then
  echo "== wasm32-wasip1 (node) =="
  RUSTFLAGS="-C target-feature=+simd128,+bulk-memory" \
    cargo build --release -q -p traverse-bench --target wasm32-wasip1
  node --no-warnings -e '
    const { WASI } = require("node:wasi");
    const fs = require("node:fs");
    const wasi = new WASI({ version: "preview1", args: ["traverse-bench", process.argv[1]], env: {} });
    const bytes = fs.readFileSync("target/wasm32-wasip1/release/traverse-bench.wasm");
    WebAssembly.instantiate(bytes, { wasi_snapshot_preview1: wasi.wasiImport }).then(({ instance }) => {
      process.exit(wasi.start(instance));
    });
  ' "$rounds"
fi

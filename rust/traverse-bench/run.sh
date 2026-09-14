#!/usr/bin/env bash
# LoD traverse 基準:原生 + wasm32-wasip1(Node 內建 WASI 跑,不用裝 wasmtime)。
# 用法:rust/traverse-bench/run.sh [native|wasm|both|walk] [rounds] [eps] [diag]
# walk 模式呼叫 lod_synth::walk_main(20 站走路模擬,方案 B Task 5),rounds 預設 1(= 20 站)、
# 原生 + wasm 都跑。eps = 遲滯帶,預設 0.05(production 值,spec D6 2026-09-14 更新;
# 例如 `run.sh walk 1 0.0` 測零遲滯,Task 5 review round 2 用來跟原子比對)。diag = "1"
# 開每站 t/cut_size/atomic 診斷輸出。native/wasm/both 三種模式維持原本的 bench_main
# (rounds 預設 5,不吃 eps/diag)。
# wasm 用與 build_rust_wasm.sh 相同的 target-feature(simd128 + bulk-memory),數字才跟瀏覽器裡的
# worker 同一種碼;Node 的 V8 與瀏覽器同一顆引擎。
# `set -e` 不含 run 那兩行(見下方 `|| status=$?`):walk 模式的 `walk_main` 尾端在
# `eps == 0.0` 時才 `assert!(jaccard >= 0.93)`(Task 5 review round 2 裁決:遲滯帶內的
# 節點依定義不動,這道 assert 只在零遲滯下才是「跟原子等價」的判準;`eps > 0` 只印
# informational 訊息)、不論 eps 為何都 `assert!(cut_size >= 0.95·max)`——皆是刻意留著的
# 可證偽品質閘門(非印出來的軟警示)。native 那支 assert 失敗不該讓 wasm 那支永遠跑不到,
# 兩邊的 WALK:/JACCARD 都要印出來才看得出是不是同一個現象。腳本結尾仍會用非 0 退出碼
# 反映「至少一支失敗」,CI 看得到。
set -uo pipefail
cd "$(dirname "$0")/.."
mode="${1:-both}"
if [[ "$mode" == "walk" ]]; then
  rounds="${2:-1}"
  bin_args=(walk "$rounds")
  [[ -n "${3:-}" ]] && bin_args+=("$3")
  # $4(diag)有給但 $3(eps)沒給時,補上跟 main.rs 相同的預設值佔位,確保 $4 落在
  # main.rs 期待的第 4 個位置(args[4] == "1" 才開 diag)。
  [[ -n "${4:-}" ]] && { [[ -z "${3:-}" ]] && bin_args+=("0.05"); bin_args+=("$4"); }
else
  rounds="${2:-5}"
  bin_args=("$rounds")
fi
native_status=0
wasm_status=0
if [[ "$mode" == "native" || "$mode" == "both" || "$mode" == "walk" ]]; then
  echo "== native =="
  cargo run --release -q -p traverse-bench -- "${bin_args[@]}" || native_status=$?
fi
if [[ "$mode" == "wasm" || "$mode" == "both" || "$mode" == "walk" ]]; then
  echo "== wasm32-wasip1 (node) =="
  RUSTFLAGS="-C target-feature=+simd128,+bulk-memory" \
    cargo build --release -q -p traverse-bench --target wasm32-wasip1 || { echo "wasm build failed" >&2; exit 1; }
  node --no-warnings -e '
    const { WASI } = require("node:wasi");
    const fs = require("node:fs");
    const wasi = new WASI({ version: "preview1", args: ["traverse-bench", ...process.argv.slice(1)], env: {} });
    const bytes = fs.readFileSync("target/wasm32-wasip1/release/traverse-bench.wasm");
    WebAssembly.instantiate(bytes, { wasi_snapshot_preview1: wasi.wasiImport }).then(({ instance }) => {
      process.exit(wasi.start(instance));
    });
  ' "${bin_args[@]}" || wasm_status=$?
fi
if [[ "$native_status" != "0" || "$wasm_status" != "0" ]]; then
  echo "!! non-zero exit (native=$native_status wasm=$wasm_status) —— 見上方 panic/assert 訊息" >&2
  exit 1
fi

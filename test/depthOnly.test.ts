import assert from "node:assert";
import { type DepthOnlyInputs, canDepthOnly } from "../src/depthOnly.js";

// 全部成立的基準:非 ext、相機離生成原點 0.5、上限 1
const ok: DepthOnlyInputs = {
  viewChanged: true,
  versionSame: true,
  mappingSame: true,
  hasTarget: true,
  extSplats: false,
  originOffset: 0.5,
  maxOffset: 1,
};
assert.strictEqual(canDepthOnly(ok), true, "基準:走快路徑");

// 四個閘門各自缺一 → 不走(缺一個 = 整組 generate)
assert.strictEqual(
  canDepthOnly({ ...ok, viewChanged: false }),
  false,
  "視角沒變 → 根本不需要重排序,交給 needsUpdate 判",
);
assert.strictEqual(
  canDepthOnly({ ...ok, versionSame: false }),
  false,
  "generator 改版(SH / LoD 索引 / updateVersion)→ 整組重建",
);
assert.strictEqual(
  canDepthOnly({ ...ok, mappingSame: false }),
  false,
  "可見集合變了 → 整組重建",
);
assert.strictEqual(
  canDepthOnly({ ...ok, hasTarget: false }),
  false,
  "第一次:current 還沒 generate 過,沒有 packed 中心可讀",
);

// 非 ext:原點偏移的上限(f16 相對中心的精度)
assert.strictEqual(
  canDepthOnly({ ...ok, originOffset: 1 }),
  true,
  "剛好等於上限 → 仍走(<=)",
);
assert.strictEqual(
  canDepthOnly({ ...ok, originOffset: 1.0001 }),
  false,
  "超過上限 → 重建,把參考原點拉回相機旁",
);
assert.strictEqual(
  canDepthOnly({ ...ok, originOffset: 0 }),
  true,
  "原地轉頭(偏移 0)→ 走",
);

// ext(f32 絕對座標):偏移不設限
assert.strictEqual(
  canDepthOnly({ ...ok, extSplats: true, originOffset: 1e6 }),
  true,
  "ext 模式任何偏移都走",
);

// maxOffset 是總開關:0 / 負數 / NaN 一律關(等同 v2.1.0 行為),ext 也關
assert.strictEqual(canDepthOnly({ ...ok, maxOffset: 0 }), false, "0 = 關");
assert.strictEqual(canDepthOnly({ ...ok, maxOffset: -1 }), false, "負數 = 關");
assert.strictEqual(
  canDepthOnly({ ...ok, maxOffset: Number.NaN }),
  false,
  "NaN = 關(不是靜默走)",
);
assert.strictEqual(
  canDepthOnly({ ...ok, extSplats: true, maxOffset: 0 }),
  false,
  "ext 模式 maxOffset 0 也關(總開關對兩種編碼都有效)",
);
assert.strictEqual(
  canDepthOnly({
    ...ok,
    maxOffset: Number.POSITIVE_INFINITY,
    originOffset: 1e9,
  }),
  true,
  "Infinity = 非 ext 也不設限",
);

console.log("✅ depthOnly test cases passed!");

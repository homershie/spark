import assert from "node:assert";
import {
  LOD_SLICE_MAX_MS,
  canAbortRound,
  newLodRound,
  shouldApplySlice,
  sliceBudget,
} from "../src/lodRound.js";

// sliceBudget:第 0 片用 first(沒給就用 sliceMs),之後 1.5× 遞增、封頂 200;sliceMs 0 = 原子
assert.strictEqual(sliceBudget(0, 40, 0), 40, "第 0 片、沒 first → sliceMs");
assert.strictEqual(sliceBudget(0, 40, 80), 80, "第 0 片、有 first → first");
assert.strictEqual(sliceBudget(1, 40, 80), 60, "第 1 片 = 40 × 1.5");
assert.strictEqual(sliceBudget(2, 40, 0), 90, "第 2 片 = 40 × 1.5²");
assert.strictEqual(sliceBudget(3, 40, 0), 135, "第 3 片 = 40 × 1.5³");
assert.strictEqual(
  sliceBudget(4, 40, 0),
  LOD_SLICE_MAX_MS,
  "40 × 1.5⁴ = 202.5 → 封頂 200",
);
assert.strictEqual(sliceBudget(50, 40, 0), LOD_SLICE_MAX_MS, "之後都 200");
assert.strictEqual(sliceBudget(0, 0, 80), 0, "sliceMs 0 = 原子,first 也不理");
assert.strictEqual(sliceBudget(3, 0, 0), 0, "sliceMs 0 任何片都 0");
assert.strictEqual(sliceBudget(0, -5, 0), 0, "負數當 0");

// canAbortRound:沒 round / 已完成 → 可;未套過任何一片 → 不可(底線);hold 內 → 不可
assert.strictEqual(canAbortRound(null, 1000, 0), true, "沒 round");
const done = { ...newLodRound(0, "init"), done: true };
assert.strictEqual(canAbortRound(done, 1, 9999), true, "已完成的輪一律可換");
const fresh = newLodRound(1000, "pose");
assert.strictEqual(
  canAbortRound(fresh, 5000, 0),
  false,
  "applied 0 → 即使 hold 0 也不可中止(底線)",
);
const applied = { ...fresh, applied: 1 };
assert.strictEqual(
  canAbortRound(applied, 1000, 0),
  true,
  "hold 0 + 套過一片 → 可",
);
assert.strictEqual(
  canAbortRound(applied, 1200, 300),
  false,
  "hold 300、才過 200ms → 不可",
);
assert.strictEqual(canAbortRound(applied, 1300, 300), true, "剛好 300ms → 可");

// newLodRound
const r = newLodRound(42, "tree", 3);
assert.deepStrictEqual(r, {
  startedAt: 42,
  slice: 0,
  applied: 0,
  done: false,
  restart: true,
  cause: "tree",
  firstApplyMs: 0,
  fetchersAtStart: 3,
});
assert.strictEqual(newLodRound(0, "init").fetchersAtStart, 0, "沒給就 0");

// shouldApplySlice:done / 原子 / 關掉規則一律套;否則顆數要 ≥ fraction × 上次套的
assert.strictEqual(
  shouldApplySlice(true, 40, 0.5, 10, 1000),
  true,
  "done 一律套",
);
assert.strictEqual(
  shouldApplySlice(false, 0, 0.5, 10, 1000),
  true,
  "budget 0(原子)一律套",
);
assert.strictEqual(
  shouldApplySlice(false, 40, 0, 10, 1000),
  true,
  "fraction 0 = 規則關,一律套",
);
assert.strictEqual(
  shouldApplySlice(false, 40, 0.5, 499, 1000),
  false,
  "499 < 500 → 不套",
);
assert.strictEqual(
  shouldApplySlice(false, 40, 0.5, 500, 1000),
  true,
  "剛好 500 → 套",
);
assert.strictEqual(
  shouldApplySlice(false, 40, 0.5, 1, 0),
  true,
  "螢幕上還沒有 cut(0)→ 任何片都套",
);

console.log("✅ lodRound tests passed!");

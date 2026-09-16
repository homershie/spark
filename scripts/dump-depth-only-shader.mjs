import { dirname, resolve } from "node:path";
// 用法:npm run build 之後 `node scripts/dump-depth-only-shader.mjs`
// 印出 depth-only 快路徑(SplatAccumulator.getDepthOnlyProgram)兩種 layout(非 ext f16 / ext f32)
// 生成的 GLSL 與 uniform 清單。改了 packed 編碼(splatDefines.glsl 的 packSplatEncoding word1/word2、
// packSplatExt x/y/z)或 outputSplatDepth 之後跑它,對照 unpackSplatEncoding / generate 的
// OutputSplatDepth 段是否仍逐字一致 —— 這段沒有 GPU 也驗得到,不用靠眼睛看透視穿幫。
import { fileURLToPath } from "node:url";

// dist 在模組載入時會摸 window / document,給最小的假物件
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.document = {
  createElement: () => ({ getContext: () => null }),
  createElementNS: () => ({}),
};

const here = dirname(fileURLToPath(import.meta.url));
const spark = await import(resolve(here, "../dist/spark.module.js"));
const A = spark.SplatAccumulator;
for (const ext of [false, true]) {
  const p = A.getDepthOnlyProgram(ext);
  console.log(`===== extSplats=${ext} =====`);
  console.log(
    p.shader
      .split("\n")
      .filter((l) => !l.startsWith("precision"))
      .join("\n"),
  );
  console.log("uniforms:", Object.keys(p.uniforms));
}

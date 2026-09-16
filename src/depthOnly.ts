// 「只重排序、不重建」快路徑的判準(純函式,`test/depthOnly.test.ts`)。
//
// SparkRenderer.updateInternal 原本只要 `viewChanged`(相機 > 1mm / > 2.6°)就整組 generate():
// 每顆可見 mesh 各跑一個 GPU pass,只為了讓 generate 順便寫出的**排序鍵**(depth attachment)
// 跟上新視角。當 accumulator 的內容沒變(version / mapping 都沒變),packed splat 本身不用重寫,
// 只需要一個便宜的 depth-only pass 從已生好的 packed 中心重算深度(`SplatAccumulator.regenerateDepth`)。
//
// 唯一的例外是**非 ext 模式的 f16 精度**:packed 中心是相對 `viewOrigin`(生成當下的相機位置)的
// float16,離參考原點越遠量化越粗(1–2 單位 0.5mm、4–8 單位 2mm、64–128 單位 3cm)。原本每次
// 視角變動都重建,參考原點永遠貼著相機;走快路徑之後相機會漸漸離開參考原點,近處的 splat 會
// 用「離原點較遠」的粗量化來畫。所以相機離生成原點超過 `maxOffset` 就退回整組重建,把參考原點
// 拉回相機旁邊。ext 模式(f32 絕對座標)沒有這個問題,不設上限。

export type DepthOnlyInputs = {
  /** 相機相對上一次排序視角動了(SparkRenderer 的 `viewChanged`)。 */
  viewChanged: boolean;
  /** `next.prepareGenerate` 算出的 version 與 `current.version` 相同 = 沒有任何 generator 改版。 */
  versionSame: boolean;
  /** mappingVersion 相同 = 可見 mesh 集合與各自的 base/count 都沒變。 */
  mappingSame: boolean;
  /** `current.target` 已配置(至少 generate 過一次;第一幀沒有東西可以重算深度)。 */
  hasTarget: boolean;
  /** accumulator 走 ext 編碼(`accumExtSplats`):中心是 f32 絕對座標,不受原點距離影響。 */
  extSplats: boolean;
  /** 相機現在的位置到 `current.viewOrigin`(packed 中心的參考原點)的距離。 */
  originOffset: number;
  /** `SparkRenderer.depthOnlyMaxOffset`:非 ext 模式允許的最大原點偏移;≤ 0 / 非數 = 整個快路徑關閉。 */
  maxOffset: number;
};

/**
 * 這一幀能不能走 depth-only 快路徑(不 generate、只重算深度再排序)。
 * 四個閘門缺一不可;`maxOffset` ≤ 0 是 A/B 的總開關(等同 v2.1.0 行為)。
 */
export function canDepthOnly({
  viewChanged,
  versionSame,
  mappingSame,
  hasTarget,
  extSplats,
  originOffset,
  maxOffset,
}: DepthOnlyInputs): boolean {
  if (!viewChanged || !versionSame || !mappingSame || !hasTarget) {
    return false;
  }
  if (!(maxOffset > 0)) {
    // 0 / 負數 / NaN:功能關閉
    return false;
  }
  if (extSplats) {
    return true;
  }
  return originOffset <= maxOffset;
}

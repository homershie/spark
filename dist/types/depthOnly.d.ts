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
export declare function canDepthOnly({ viewChanged, versionSame, mappingSame, hasTarget, extSplats, originOffset, maxOffset, }: DepthOnlyInputs): boolean;

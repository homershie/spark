/**
 * LoD traverse 時間切片的 round 策略(純函式,`test/lodRound.test.ts` 有測試)。
 *
 * 一個 round = 從「姿態變了、從 root 重開」到「跑完」;每幀跑一片(slice)、套用快照。
 * 設計與決策理由見 spark-react-r3f 專案的
 * `docs/superpowers/specs/2026-09-11-sliced-traverse-design.md` §5 與 D5/D6。
 */
/** 片預算遞增比(第 k 片 = sliceMs × GROWTH^k)。 */
export declare const LOD_SLICE_GROWTH = 1.5;
/** 片預算上限(ms)。 */
export declare const LOD_SLICE_MAX_MS = 200;
export type LodRoundCause = "init" | "pose" | "tree";
export type LodRound = {
    /** performance.now() 起點。 */
    startedAt: number;
    /** 已跑過幾片(下一片的索引)。 */
    slice: number;
    /** 已套用到 GPU 幾片。 */
    applied: number;
    done: boolean;
    /** 下一次呼叫要不要叫 Rust 從 root 重開。 */
    restart: boolean;
    cause: LodRoundCause;
    /** 第一片套用時距 startedAt 的毫秒(0 = 還沒套過)。 */
    firstApplyMs: number;
    /** round 開始時在飛的 chunk fetcher 數(讀數用:串流忙 / 靜止分群)。 */
    fetchersAtStart: number;
};
export type LodRoundStats = {
    firstApplyMs: number;
    totalMs: number;
    slices: number;
    aborted: boolean;
    cause: LodRoundCause;
    fetchersAtStart: number;
};
/**
 * 第 `slice` 片的預算(ms)。`sliceMs <= 0` = 原子模式,回 0。
 * 第 0 片用 `firstMs`(> 0 時),否則 `sliceMs`;之後 1.5× 遞增、封頂 200 ——
 * 固定 40ms 要 28 片、pack + 上傳累計 +25% 會踩破「不劣化 20%」;遞增只要 ~8 片(+7%)。
 */
export declare function sliceBudget(slice: number, sliceMs: number, firstMs: number): number;
/**
 * 姿態變了,現在這輪能不能丟掉重開。
 * 底線:**至少套用過一片**才能被中止 —— 否則 `firstMs` 大於姿態變髒的間隔時,
 * 永遠沒有一片套得上、畫面凍在舊 cut。`holdMs` 是額外的最短存活時間。
 */
export declare function canAbortRound(round: LodRound | null, now: number, holdMs: number): boolean;
export declare function newLodRound(now: number, cause: LodRoundCause, fetchersAtStart?: number): LodRound;

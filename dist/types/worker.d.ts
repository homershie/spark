/** 增量 tick(方案 B)的讀數;原子路徑(`incremental=false` 或 `budgetMs≤0`)不回這個物件。 */
export type LodTickStats = {
    scanned: number;
    expanded: number;
    collapsed: number;
    passes: number;
    settled: boolean;
    evicted: number;
    wantedCount: number;
    cutSize: number;
    t: number;
    arenaLeaked: number;
    changed: boolean;
    tickMs: number;
    boundSkipped: number;
    /** D23:這個 tick 把幾個「等頁到達」的拆/收候選直接推回 heap(不等掃描重新找到)。 */
    requeued: number;
    /** D22:這次呼叫有沒有交出 `instanceIndices`(= `packNow && needs_pack()`)。 */
    packed: boolean;
    /**
     * D22:這次呼叫**之後** cut 是否仍與上一次交出去的 pack 不同(剛 pack 過 → false)。
     * true = 螢幕上那份已過期,`fetchPriority` 要連它用到的 chunks 一起保住(不變式 2)。
     */
    needsPack: boolean;
};

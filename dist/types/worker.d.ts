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
};

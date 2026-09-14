import { ExtSplats, PackedSplats, PagedSplats, SplatMesh, SplatPager } from '.';
import { SplatAccumulator } from './SplatAccumulator';
import { SplatWorker } from './SplatWorker';
import { LodTickStats } from './worker';
import * as THREE from "three";
export interface SparkRendererOptions {
    /**
     * Pass in your THREE.WebGLRenderer instance so Spark can perform work
     * outside the usual render loop. Should be created with antialias: false
     * (default setting) as WebGL anti-aliasing doesn't improve Gaussian Splatting
     * rendering and significantly reduces performance.
     */
    renderer: THREE.WebGLRenderer;
    /**
     * Callback function to be called when SparkRenderer needs to re-render,
     * for example when splat sort order or LoD updates complete.
     */
    onDirty?: () => void;
    /**
     * Whether to use premultiplied alpha when accumulating splat RGB
     * @default true
     */
    premultipliedAlpha?: boolean;
    /**
     * Whether to encode Gsplat with linear RGB (for environment mapping)
     * @default false
     */
    encodeLinear?: boolean;
    /**
     * Pass in a THREE.Clock to synchronize time-based effects across different
     * systems. Alternatively, you can set the property time directly.
     * (default: new THREE.Clock)
     */
    clock?: THREE.Clock;
    /**
     * Controls whether to check and automatically update Gsplat collection
     * each frame render.
     * @default true
     */
    autoUpdate?: boolean;
    /**
     * Controls whether to update the Gsplats before or after rendering. For WebXR
     * this is set to false in order to complete rendering as soon as possible.
     * @default true (if not WebXR)
     */
    preUpdate?: boolean;
    /**
     * Maximum standard deviations from the center to render Gaussians. Values
     * Math.sqrt(4)..Math.sqrt(9) produce acceptable results and can be tweaked for
     * performance.
     * @default Math.sqrt(8)
     */
    maxStdDev?: number;
    minPixelRadius?: number;
    /**
     * Maximum pixel radius for splat rendering.
     * @default 512.0
     */
    maxPixelRadius?: number;
    /**
     * Whether to use extended Gsplat encoding for intermediary accumulator splats.
     * @default false
     */
    accumExtSplats?: boolean;
    /**
     * Whether to use covariance Gsplat encoding for intermediary splats.
     * @default false
     */
    covSplats?: boolean;
    /**
     * Minimum alpha value for splat rendering.
     * @default 0.5 * (1.0 / 255.0)
     */
    minAlpha?: number;
    /**
     * Enable 2D Gaussian splatting rendering ability. When this mode is enabled,
     * any scale x/y/z component that is exactly 0 (minimum quantized value) results
     * in the other two non-0 axis being interpreted as an oriented 2D Gaussian Splat,
     * rather instead of the usual projected 3DGS Z-slice. When reading PLY files,
     * scale values less than e^-30 will be interpreted as 0.
     * @default false
     */
    enable2DGS?: boolean;
    /**
     * Enable alternative ray-splat max response evaluation, used by 3DGUT (unscented transform),
     * 3DGRT, and HTGS.
     * @default false
     */
    /**
     * Scalar value to add to 2D splat covariance diagonal, effectively blurring +
     * enlarging splats. In scenes trained without the Gsplat anti-aliasing tweak
     * this value was typically 0.3, but with anti-aliasing it is 0.0
     * @default 0.0
     */
    preBlurAmount?: number;
    /**
     * Scalar value to add to 2D splat covarianve diagonal, with opacity adjustment
     * to correctly account for "blurring" when anti-aliasing. Typically 0.3
     * (equivalent to approx 0.5 pixel radius) in scenes trained with anti-aliasing.
     */
    blurAmount?: number;
    /**
     * Depth-of-field distance to focal plane
     */
    focalDistance?: number;
    /**
     * Full-width angle of aperture opening (in radians), 0.0 to disable
     * @default 0.0
     */
    apertureAngle?: number;
    /**
     * Modulate Gaussian kernel falloff. 0 means "no falloff, flat shading",
     * while 1 is the normal Gaussian kernel.
     * @default 1.0
     */
    falloff?: number;
    /**
     * X/Y clipping boundary factor for Gsplat centers against view frustum.
     * 1.0 clips any centers that are exactly out of bounds, while 1.4 clips
     * centers that are 40% beyond the bounds.
     * @default 1.4
     */
    clipXY?: number;
    /**
     * Parameter to adjust projected splat scale calculation to match other renderers,
     * similar to the same parameter in the MKellogg 3DGS renderer. Higher values will
     * tend to sharpen the splats. A value 2.0 can be used to match the behavior of
     * the PlayCanvas renderer.
     * @default 1.0
     */
    focalAdjustment?: number;
    /**
     * Whether to sort splats radially (geometric distance) from the viewpoint (true)
     * or by Z-depth (false). Most scenes are trained with the Z-depth `sort `metric
     * and will render more accurately at certain viewpoints. However, radial sorting
     * is more stable under viewpoint rotations.
     * @default true
     */
    sortRadial?: boolean;
    /**
     * Minimum interval between sort calls in milliseconds.
     * @default 0
     */
    minSortIntervalMs?: number;
    enableLod?: boolean;
    /**
     * Whether to drive LOD updates (compute lodInstances, update pager, etc.).
     * Set to false to use LOD instances from another renderer without driving updates.
     * Only has effect if enableLod is true.
     * @default true (if enableLod is true)
     */
    enableDriveLod?: boolean;
    /**
     * Whether to enable page fetching for LoD.
     * @default true
     */
    enableLodFetching?: boolean;
    /**
     * Set the target # splats for LoD. If this isn't set then default base LoD splat
     * counts will apply: 500K-750K for WebXR, 1-1.5M for mobile, and 2.5M for desktop.
     * @default 500K-2500K depending on platform
     */
    lodSplatCount?: number;
    /**
     * Scale factor for target # splats for LoD. 2.0 means 2x the base LoD splat count.
     * This is the easiest LoD parameter to adjust and will scale detail appropriately
     * for the platform.
     * @default 1.0
     */
    lodSplatScale?: number;
    /**
     * Determines the minimum screen pixel size of LoD splats. The default 1.0 means
     * the splat LoD tree will pick splats that are no smaller than 1 pixel in size.
     * Setting this to a higher value as high as 5.0 will often be indistinguishable
     * but will avoid wasting rendering capacity on tiny splats.
     * @default 1.0
     */
    lodRenderScale?: number;
    /**
     * LoD cut 走增量(方案 B):保留上一幀的展開樹、姿態變了只局部拆 / 收,每 tick(一次 worker
     * 呼叫)交一次合法 cut。false = 每次姿態變動從 root 原子重走(v2.1.0 行為,A/B 對照)。
     * 設計:spark-react-r3f `docs/superpowers/specs/2026-09-11-incremental-traverse-design.md`。
     * @default true
     */
    lodIncremental?: boolean;
    /**
     * 每 tick 的 wasm 時間預算(ms):掃描 + 拆收,pack 不計。
     * ≤ 0 = 每次髒了從 root 原子重走(等同 lodIncremental=false)。
     */
    lodTickMs?: number;
    /** 拆 / 收的遲滯 ε:拆要 ps > t·(1+ε)、收要 ps ≤ t·(1−ε),壓住門檻邊的閃爍。 */
    lodHysteresis?: number;
    /**
     * 兩次 GPU 索引套用的最短間隔(ms);0 = 每 tick 套。D22:到期前的 tick 連 worker 端的
     * pack 都省掉(`packNow=false`),不只是延後上傳。
     */
    lodApplyIntervalMs?: number;
    /**
     * Inflate LoD splats to ensure opacity stays <= 1.0, producing a softer appearance.
     * @default false
     */
    lodInflate?: boolean;
    /**
     * Whether to use extended Gsplat encoding for paged splats, useful for eliminating
     * quantization artifacts from splat scenes with large internal position coordinates.
     * @default false
     */
    pagedExtSplats?: boolean;
    /**
     * Allocation size of paged splats. This must be a multiple of the page size (65536).
     * @default 16777216 (256 * 65536) for desktop, 6291456 for iOS, 8,388,608 for other mobile
     */
    maxPagedSplats?: number;
    /**
     * Number of parallel chunk fetchers for LoD. These are run within a shared pool
     * of 4 background WebWorker threads, so setting it above 4 will not have any
     * effect. Setting it 3 leaves one spare worker for other loading/decoding tasks.
     * @default 3
     */
    numLodFetchers?: number;
    /**
     * Full-width angle in degrees of fixed foveation cone along the view direction
     * with no foveation applied (full resolution, foveate=1.0). Set to 0 to disable.
     * @default 90.0
     */
    coneFov0?: number;
    /**
     * Full-width angle in degrees of fixed foveation cone along the view direction
     * with reduced resolution specified by `coneFoveate`. Foveation will be applied
     * smoothly from 1.0 down to `coneFoveate` as you move outward from
     * `coneFov0` to `coneFov`. Set to 0 to disable.
     * @default 120.0
     */
    coneFov?: number;
    /**
     * Foveation scale to apply to LoD splats at the edge of coneFov. Foveation will
     * be applied smoothly from `coneFoveate` down to `behindFoveate` as you move
     * outward from `coneFov` to 180 degrees (behind the viewer).
     * @default 0.4
     */
    coneFoveate?: number;
    /**
     * Foveation scale to apply to LoD splats behind the viewer. Setting this to 0.1
     * for example will result in splats 10x larger than inside the viewing frustum.
     * @default 0.2
     */
    behindFoveate?: number;
    /**
     * How many LoD splats to generate for raycasting
     * @default 10000-25000 iff default canvas target is used
     */
    lodRaycast?: number;
    lodRaycastIntervalMs?: number;
    /**
     * Configures an offline render target for the SparkRenderer (as opposed to
     * rendering to the canvas). This is useful for rendering environment maps,
     * additional viewpoints, or video frame rendering.
     * @default undefined
     */
    target?: {
        /**
         * Width of the render target in pixels.
         */
        width: number;
        /**
         * Height of the render target in pixels.
         */
        height: number;
        /**
         * If you want to be able to render a scene that depends on this target's
         * output (for example, a recursive viewport), set this to true to enable
         * double buffering.
         * @default false
         */
        doubleBuffer?: boolean;
        /**
         * Super-sampling factor for the render target. Values 1-4 are supported.
         * Note that re-sampling back down to .width x .height is done on the CPU
         * with simple averaging only when calling readTarget().
         * @default 1
         */
        superXY?: number;
    } & THREE.RenderTargetOptions;
    /**
     * Extra uniform values to pass to the shader.
     * @default undefined = no extra uniforms
     */
    extraUniforms?: Record<string, unknown>;
    /**
     * Replace the default `splatVertex.glsl` splat shader with a custom one.
     * @default undefined = use the default `splatVertex.glsl` shader
     */
    vertexShader?: string;
    /**
     * Replace the default `splatFragment.glsl` splat shader with a custom one.
     * @default undefined = use the default `splatFragment.glsl` shader
     */
    fragmentShader?: string;
    /**
     * Set the splat shader material to be transparent which determines if the
     * splats are rendered during the first opaque THREE.js render pass or the
     * second transparent render pass.
     * @default undefined = true
     */
    transparent?: boolean;
    /**
     * Set the splat shader material to enable depth testing which determines if the
     * splats respect the Z depth buffer and blend with other opaque objects in the scene.
     * @default undefined = true
     */
    depthTest?: boolean;
    /**
     * Set the splat shader material to enable depth writing which determines if the
     * splats write to the Z depth buffer. Note that enabling this may produce
     * undesirable results because most of the Gsplat is transparent.
     * @default undefined = false
     */
    depthWrite?: boolean;
}
export declare class SparkRenderer extends THREE.Mesh {
    readonly renderer: THREE.WebGLRenderer;
    readonly material: THREE.ShaderMaterial;
    readonly uniforms: ReturnType<typeof SparkRenderer.makeUniforms>;
    autoUpdate: boolean;
    preUpdate: boolean;
    static sparkOverride?: SparkRenderer;
    renderSize: THREE.Vector2;
    maxStdDev: number;
    minPixelRadius: number;
    maxPixelRadius: number;
    accumExtSplats: boolean;
    covSplats: boolean;
    minAlpha: number;
    enable2DGS: boolean;
    preBlurAmount: number;
    blurAmount: number;
    focalDistance: number;
    apertureAngle: number;
    falloff: number;
    clipXY: number;
    focalAdjustment: number;
    encodeLinear: boolean;
    sortRadial: boolean;
    minSortIntervalMs: number;
    clock: THREE.Clock;
    time?: number;
    lastFrame: number;
    updateTimeoutId: number;
    onDirty?: () => void;
    dirty: boolean;
    orderingTexture: THREE.DataTexture | null;
    maxSplats: number;
    activeSplats: number;
    display: SplatAccumulator;
    current: SplatAccumulator;
    accumulators: SplatAccumulator[];
    sorting: boolean;
    sortDirty: boolean;
    lastSortTime: number;
    sortWorker: SplatWorker | null;
    sortTimeoutId: number;
    sortedCenter: THREE.Vector3;
    sortedDir: THREE.Vector3;
    readback32: Uint32Array<ArrayBuffer>;
    enableLod: boolean;
    enableDriveLod: boolean;
    enableLodFetching: boolean;
    lodSplatCount?: number;
    lodSplatScale: number;
    lodRenderScale: number;
    /**
     * LoD cut 走增量(方案 B):保留上一幀的展開樹、姿態變了只局部拆 / 收。false = 每次姿態
     * 變動從 root 原子重走(v2.1.0 行為,A/B 對照)。
     */
    lodIncremental: boolean;
    /**
     * 每 tick 的 wasm 時間預算(ms):掃描 + 拆收,pack 不計。
     * ≤ 0 = 每次髒了從 root 原子重走(等同 lodIncremental=false)。
     */
    lodTickMs: number;
    /** 拆 / 收的遲滯 ε:拆要 ps > t·(1+ε)、收要 ps ≤ t·(1−ε),壓住門檻邊的閃爍。 */
    lodHysteresis: number;
    /**
     * 兩次 GPU 索引套用的最短間隔(ms);0 = 每 tick 套。D22:到期前的 tick 連 worker 端的
     * pack 都省掉(`packNow=false`),不只是延後上傳。
     */
    lodApplyIntervalMs: number;
    /** 最近一次 tick 的讀數;null = 還沒 tick 過。 */
    lastLodTick: (LodTickStats & {
        neededChunks: number;
        pagePressure: boolean;
    }) | null;
    /** 每完成一次 tick(`lastLodTick` 被更新)+1;讀數的邊緣訊號。 */
    lodTickSeq: number;
    /** 最近一次 pose 髒 → settled 的毫秒;還沒 settled 期間是 null。 */
    lodSettleMs: number | null;
    /** 最近一次姿態變髒的 performance.now();用於量測 lodSettleMs。 */
    private lodPoseDirtyAt;
    /**
     * 上一次「要 worker pack、並把回來的索引套進 GPU」的 performance.now()(節流用)。
     * D22 的契約:節流不再是「worker 每次都 pack、JS 把多餘的暫存」,而是**JS 到期才叫 worker
     * pack**(`packNow`),回來的索引一律立刻套——沒有 pending 這一層了。
     */
    private lodLastApplyAt;
    /**
     * 目前**螢幕上顯示**那份 cut 帶的 chunks。兩次套用之間 worker 仍在拆收,最新 tick 的 `chunks`
     * 是「最新表」的投影,可能已經不含螢幕上那份用到的某個 chunk;`lodCutStale` 為 true 時
     * fetchPriority 得把這份排最前面保住(spec §4.5 不變式 2)。
     */
    private lodAppliedChunks;
    /**
     * D22:最近一次 tick 回報的 `needsPack`——true = cut 已經跟螢幕上那份(上一次交出去的 pack)
     * 不同、還沒重 pack。原子路徑恆 false。
     */
    private lodCutStale;
    /** 頁面更新到了,會讓下面的 needTick 在這一幀強制 tick 一次(不等 settled);見 driveLod。 */
    lodTreeDirty: boolean;
    lodInflate: boolean;
    pagedExtSplats: boolean;
    maxPagedSplats: number;
    numLodFetchers: number;
    behindFoveate: number;
    coneFov0: number;
    coneFov: number;
    coneFoveate: number;
    lodRaycast?: number;
    lodRaycastIntervalMs: number;
    lastLodRaycastTime: number;
    lodWorker: SplatWorker | null;
    /**
     * 上一幀參與 LoD 的 mesh 與「預期的 version」;driveLod 用 `mesh.version > 記錄值` 判髒。
     * ⚠️ `updateLodIndices` 套用一次索引時自己會 bump version(+1,下一幀 accumulator 看到
     * numSplats 變了再 +1),那裡會把記錄提到 `mesh.version + 1` 讓自己的 bump 不算髒 ——
     * 否則每套用一次就自己標髒、needTick 永遠是 true。附帶效果:原子模式(`lodIncremental = false`)
     * 站著不動時也不再每幀無止境重跑 traverse(v2.1.0 是會的)。
     */
    lodMeshes: {
        mesh: SplatMesh;
        version: number;
    }[];
    lodDirty: boolean;
    /**
     * 診斷讀數:每個「把 lodDirty 標髒」的來源各累計幾次(自頁面載入起,不歸零)。
     * key:pixelScaleLimit / maxSplats / pose / meshCount / meshVersion / init /
     * external(進 driveLod 時已經是 true = 外部寫的)。
     */
    lodDirtyReasons: Record<string, number>;
    /** 最近一次 pose 比對量到的位移與四元數 dot(每幀寫,看姿態到底動了多遠)。 */
    lodLastPoseDelta: {
        distance: number;
        dot: number;
    };
    /**
     * `markLodDirty("pose")` **當下**那一幀的完整快照(每幀寫的 lodLastPoseDelta 會被
     * 後面乾淨的幀蓋掉,看不到觸發那一幀)。陣列都是 plain number,可直接 JSON。
     */
    lodLastPoseDirty: {
        distance: number;
        dot: number;
        hadPosOverride: boolean;
        hadQuatOverride: boolean;
        viewPos: number[];
        lastPos: number[];
        viewQuat: number[];
        lastQuat: number[];
        frame: number;
        camera: string;
        renderSizeY: number;
    };
    /** pose 標髒是哪條 ramp 造成的(累計):距離 ≥ 0.001 / dot < 0.99999 / 兩者。 */
    lodPoseDirtyHist: {
        dotBelow: number;
        distAbove: number;
        both: number;
    };
    /** 每次 driveLod 的 camera(`type:uuid前8碼`)累計次數 —— 看是不是不只一顆相機在驅動。 */
    lodDriveCameras: Record<string, number>;
    /** 這一幀參與 LoD 的 mesh 數(= lodMeshes.length)。 */
    lodLastLodMeshes: number;
    /** lodDirty 目前這個 true 是不是 driveLod 自己標的(false 且 lodDirty=true ⇒ 外部寫的)。 */
    private lodDirtyOwned;
    lodIds: Map<PackedSplats | ExtSplats | PagedSplats, {
        lodId: number;
        lastTouched: number;
        rootPage?: number;
    }>;
    lodIdToSplats: Map<number, PackedSplats | ExtSplats | PagedSplats>;
    lodInitQueue: (PackedSplats | ExtSplats | PagedSplats)[];
    lastLod?: {
        pos: THREE.Vector3;
        quat: THREE.Quaternion;
        pixelScaleLimit: number;
        maxSplats: number;
        timestamp: number;
    };
    currentLod?: {
        pos: THREE.Vector3;
        quat: THREE.Quaternion;
        pixelScaleLimit: number;
        maxSplats: number;
        timestamp: number;
    };
    lodPosOverride?: THREE.Vector3;
    lodQuatOverride?: THREE.Quaternion;
    lodInstances: Map<SplatMesh, {
        lodId: number;
        numSplats: number;
        indices: Uint32Array;
        texture: THREE.DataTexture;
    }>;
    lodUpdates: {
        lodId: number;
        pageBase: number;
        chunkBase: number;
        count: number;
        lodTreeData?: Uint32Array;
    }[];
    lastTraverseTime: number;
    lastPixelLimit?: number;
    pager?: SplatPager;
    pagerId: number;
    target?: THREE.WebGLRenderTarget;
    backTarget?: THREE.WebGLRenderTarget;
    superPixels?: Uint8Array;
    targetPixels?: Uint8Array;
    superXY: number;
    flushAfterGenerate: boolean;
    flushAfterRead: boolean;
    readPause: number;
    sortPause: number;
    sortDelay: number;
    constructor(options: SparkRendererOptions);
    static makeUniforms(): {
        renderSize: {
            value: THREE.Vector2;
        };
        near: {
            value: number;
        };
        far: {
            value: number;
        };
        renderToViewQuat: {
            value: THREE.Quaternion;
        };
        renderToViewPos: {
            value: THREE.Vector3;
        };
        renderToViewBasis: {
            value: THREE.Matrix3;
        };
        renderToViewOffset: {
            value: THREE.Vector3;
        };
        maxStdDev: {
            value: number;
        };
        minPixelRadius: {
            value: number;
        };
        maxPixelRadius: {
            value: number;
        };
        minAlpha: {
            value: number;
        };
        enable2DGS: {
            value: boolean;
        };
        lodInflate: {
            value: boolean;
        };
        preBlurAmount: {
            value: number;
        };
        blurAmount: {
            value: number;
        };
        focalDistance: {
            value: number;
        };
        apertureAngle: {
            value: number;
        };
        falloff: {
            value: number;
        };
        clipXY: {
            value: number;
        };
        focalAdjustment: {
            value: number;
        };
        encodeLinear: {
            value: boolean;
        };
        ordering: {
            type: string;
            value: THREE.DataTexture;
        };
        enableExtSplats: {
            value: boolean;
        };
        enableCovSplats: {
            value: boolean;
        };
        extSplats: {
            type: string;
            value: THREE.DataArrayTexture;
        };
        extSplats2: {
            type: string;
            value: THREE.DataArrayTexture;
        };
        time: {
            value: number;
        };
        deltaTime: {
            value: number;
        };
        debugFlag: {
            value: boolean;
        };
    };
    dispose(): void;
    setDirty(): void;
    onBeforeRender(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera): void;
    clearSplats(): void;
    update({ scene, camera, }: {
        scene: THREE.Scene;
        camera: THREE.Camera;
    }): Promise<void>;
    private updateInternal;
    private driveSort;
    private ensureLodWorker;
    defaultSplatTarget(): 500000 | 750000 | 1000000 | 1500000 | 2500000;
    private driveLod;
    private bumpLodDirtyReason;
    /** driveLod 內部標髒的唯一入口:記次數 + 記「這次是我們自己標的」(區分 external)。 */
    private markLodDirty;
    private initLodTree;
    private pageSizeWarning;
    private updateLodInstances;
    private cleanupLodTrees;
    private updateLodIndices;
    private readbackDepth;
    private saveRenderState;
    private resetRenderState;
    private static emptyOrdering;
    render(scene: THREE.Scene, camera: THREE.Camera): void;
    renderTarget({ scene, camera, }: {
        scene: THREE.Scene;
        camera: THREE.Camera;
    }): THREE.WebGLRenderTarget;
    readTarget(): Promise<Uint8Array>;
    renderReadTarget({ scene, camera, }: {
        scene: THREE.Scene;
        camera: THREE.Camera;
    }): Promise<Uint8Array>;
    private static cubeRender;
    private static pmrem;
    renderCubeMap({ scene, worldCenter, size, near, far, hideObjects, update, filter, }: {
        scene: THREE.Scene;
        worldCenter: THREE.Vector3;
        size?: number;
        near?: number;
        far?: number;
        hideObjects: THREE.Object3D[];
        update: boolean;
        filter: boolean;
    }): Promise<THREE.CubeTexture>;
    readCubeTargets(): Promise<Uint8Array[]>;
    renderEnvMap({ scene, worldCenter, size, near, far, hideObjects, update, }: {
        scene: THREE.Scene;
        worldCenter: THREE.Vector3;
        size?: number;
        near?: number;
        far?: number;
        hideObjects: THREE.Object3D[];
        update: boolean;
    }): Promise<THREE.Texture>;
    recurseSetEnvMap(root: THREE.Object3D, envMap: THREE.Texture): void;
    getLodTreeLevel(splats: SplatMesh, level: number, pageColoring?: boolean): Promise<SplatMesh | null>;
    get premultipliedAlpha(): boolean;
    set premultipliedAlpha(value: boolean);
}

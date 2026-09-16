precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;
precision highp isampler2D;
precision highp sampler2DArray;
precision highp usampler2DArray;
precision highp isampler2DArray;
precision highp sampler3D;
precision highp usampler3D;
precision highp isampler3D;

#include <splatDefines>

uniform uint targetLayer;
uniform int targetBase;
uniform int targetCount;

// Depth-only pass (SplatAccumulator.regenerateDepth): the single attachment is
// named target3 so the very same outputSplatDepth dyno that generate() uses can
// write it unchanged (same encoding the sort worker expects).
layout(location = 0) out vec4 target3;

{{ GLOBALS }}

void produceDepth(int _index) {
    {{ STATEMENTS }}
}

void main() {
    int targetIndex = int(targetLayer << SPLAT_TEX_LAYER_BITS) + int(uint(gl_FragCoord.y) << SPLAT_TEX_WIDTH_BITS) + int(gl_FragCoord.x);
    int index = targetIndex - targetBase;

    // Initialize target3 to +infinity (= inactive splat, excluded by the sort)
    target3 = floatToVec4(1.0 / 0.0);

    if ((index >= 0) && (index < targetCount)) {
        produceDepth(index);
    }
}

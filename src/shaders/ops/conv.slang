module conv;

__exported import ops.matmul_simd;

[[vk::constant_id(5)]] public const uint MODE = 0;
[[vk::constant_id(6)]] public const uint DIRECT = 0;

public struct Metadata
{
    public uint64_t params;
}

// the metadata's params must match conv_metadata in conv.cpp
public struct ConvPushConstants
{
    public uint64_t x;
    public uint64_t weight;
    public uint64_t y;
    public uint64_t bias;
    public uint64_t metadata;
    public uint n;
    public uint splits;
    public uint chunk;
}

static const int PARAMS = 60;
groupshared uint params[PARAMS];

// ATen's IntDivider, exact for n below 2^31
struct Divider
{
    uint d;
    uint multiplier;
    uint shift;

    __init(int at)
    {
        d = params[at];
        multiplier = params[at + 1];
        shift = params[at + 2];
    }

    uint div(uint n) { return uint((uint64_t(uint((uint64_t(n) * multiplier) >> 32)) + n) >> shift); }
}

struct Size3
{
    int3 size;
    Divider y;
    Divider z;

    __init(int at)
    {
        size = int3(params[at], params[at + 1], params[at + 2]);
        y = Divider(at + 3);
        z = Divider(at + 6);
    }

    int3 unflatten(uint index)
    {
        uint rest = z.div(index);
        uint first = y.div(rest);
        return int3(first, rest - first * size.y, index - rest * size.z);
    }

    uint flatten(int3 at) { return uint((at.x * size.y + at.y) * size.z + at.z); }
}

struct Geometry
{
    Size3 inputSize;
    Size3 outputSize;
    Size3 kernelSize;
    int3 stride;
    int3 padding;
    int3 dilation;
    Divider strideX;
    Divider strideY;
    Divider strideZ;
    Divider inputs;
    Divider outputs;
    Divider taps;
    uint groups;
    Divider groupsDiv;
    uint cx;
    uint cy;

    __init()
    {
        inputSize = Size3(0);
        outputSize = Size3(9);
        kernelSize = Size3(18);
        stride = int3(params[27], params[28], params[29]);
        padding = int3(params[30], params[31], params[32]);
        dilation = int3(params[33], params[34], params[35]);
        groups = params[36];
        cx = params[37];
        cy = params[38];
        strideX = Divider(39);
        strideY = Divider(42);
        strideZ = Divider(45);
        inputs = Divider(48);
        outputs = Divider(51);
        taps = Divider(54);
        groupsDiv = Divider(57);
    }

    int3 strided(int3 shifted)
    {
        uint3 q = uint3(strideX.div(uint(shifted.x)), strideY.div(uint(shifted.y)), strideZ.div(uint(shifted.z)));
        return all(int3(q) * stride == shifted) ? int3(q) : int3(-1);
    }
}

T.Accum unfoldedInput<T : ITorchVulkan>(Geometry g, uint64_t x, uint group, uint c, uint tap, uint item, uint o)
{
    int3 at = g.outputSize.unflatten(o) * g.stride - g.padding + g.kernelSize.unflatten(tap) * g.dilation;
    if (any(at < int3(0)) || any(at >= g.inputSize.size)) return T.Accum(0);
    uint channel = (item * g.groups + group) * g.cx + c;
    return ((T*)x)[channel * g.inputs.d + g.inputSize.flatten(at)].toAccum();
}

struct WeightRows<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t weight;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        return ((T*)weight)[(group * g.cy + row) * g.cx * g.taps.d + col].toAccum();
    }
}

struct UnfoldedColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t x;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        uint c = g.taps.div(row);
        uint item = g.outputs.div(col);
        return unfoldedInput<T>(g, x, group, c, row - c * g.taps.d, item, col - item * g.outputs.d);
    }
}

struct WeightColumns<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t weight;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        uint m = g.taps.div(col);
        return ((T*)weight)[((group * g.cy + m) * g.cx + row) * g.taps.d + col - m * g.taps.d].toAccum();
    }
}

struct GatheredOutput<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t y;
    Geometry g;

    T.Accum load(uint group, uint row, uint col)
    {
        uint m = g.taps.div(row);
        uint item = g.inputs.div(col);
        int3 shifted = g.inputSize.unflatten(col - item * g.inputs.d) + g.padding - g.kernelSize.unflatten(row - m * g.taps.d) * g.dilation;
        if (any(shifted < int3(0))) return T.Accum(0);
        int3 o = g.strided(shifted);
        if (any(o < int3(0)) || any(o >= g.outputSize.size)) return T.Accum(0);
        uint channel = (item * g.groups + group) * g.cy + m;
        return ((T*)y)[channel * g.outputs.d + g.outputSize.flatten(o)].toAccum();
    }
}

struct OutputRows<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t y;
    Geometry g;
    uint n;
    uint chunk;

    T.Accum load(uint batch, uint row, uint col)
    {
        uint split = g.groupsDiv.div(batch);
        uint k = split * chunk + col;
        if (k >= n * g.outputs.d) return T.Accum(0);
        uint item = g.outputs.div(k);
        uint channel = (item * g.groups + batch - split * g.groups) * g.cy + row;
        return ((T*)y)[channel * g.outputs.d + k - item * g.outputs.d].toAccum();
    }
}

struct UnfoldedRows<T : ITorchVulkan> : IMatmulLoader<T.Accum>
{
    uint64_t x;
    Geometry g;
    uint n;
    uint chunk;

    T.Accum load(uint batch, uint row, uint col)
    {
        uint split = g.groupsDiv.div(batch);
        uint k = split * chunk + row;
        if (k >= n * g.outputs.d) return T.Accum(0);
        uint c = g.taps.div(col);
        uint item = g.outputs.div(k);
        return unfoldedInput<T>(g, x, batch - split * g.groups, c, col - c * g.taps.d, item, k - item * g.outputs.d);
    }
}

struct GroupedStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    uint64_t address;
    uint groups;
    uint channels;
    Divider positions;

    void store(uint group, uint row, uint col, T.Accum value)
    {
        uint item = positions.div(col);
        uint channel = (item * groups + group) * channels + row;
        ((T*)address)[channel * positions.d + col - item * positions.d] = T.fromAccum(value);
    }
}

struct BlockStore<T : ITorchVulkan> : IMatmulStore<T.Accum>
{
    uint64_t address;
    uint rows;
    uint columns;

    void store(uint batch, uint row, uint col, T.Accum value)
    {
        ((T.Accum*)address)[(batch * rows + row) * columns + col] = value;
    }
}

struct BiasEpilogue<T : ITorchVulkan> : IMatmulEpilogue<T.Accum>
{
    uint64_t bias;
    uint rows;

    T.Accum apply(uint group, uint row, uint col, T.Accum acc)
    {
        if (HAS_BIAS == 0) return acc;
        return acc + ((T*)bias)[group * rows + row].toAccum();
    }
}

public static const int DIRECT_ROWS = 8;
public static const int DIRECT_COLUMNS = 8;

void directDot<T : __BuiltinArithmeticType, A : IMatmulLoader<T>, B : IMatmulLoader<T>, E : IMatmulEpilogue<T>, S : IMatmulStore<T>>(
    uint batch, uint row0, uint col, A a, B b, E epilogue, S out, uint M, uint N, uint K)
{
    if (col >= N) return;
    uint rows = min(M - row0, uint(DIRECT_ROWS));
    T acc[DIRECT_ROWS];
    [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++) acc[r] = T(0);
    for (uint k = 0; k < K; k++)
    {
        T value = b.load(batch, k, col);
        [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++)
        {
            if (r < rows) acc[r] += a.load(batch, row0 + r, k) * value;
        }
    }
    [ForceUnroll] for (int r = 0; r < DIRECT_ROWS; r++)
    {
        if (r < rows) out.store(batch, row0 + r, col, epilogue.apply(batch, row0 + r, col, acc[r]));
    }
}

void runGemm<T : __BuiltinArithmeticType, let VEC : int, let TILE_M : int,
    A : IMatmulLoader<T>, B : IMatmulLoader<T>, E : IMatmulEpilogue<T>, S : IMatmulStore<T>>(
    uint3 li, uint3 wg, __ref T smemA[TILE_M * (TILE_K + 1)], __ref vector<T, VEC> smemB[TILE_K * (TILE_N / VEC)],
    A a, B b, E epilogue, S out, uint M, uint N, uint K)
{
    if (DIRECT == 1)
    {
        uint threads = WG_x * WG_y;
        uint col = wg.x * threads * DIRECT_COLUMNS + li.y * WG_x + li.x;
        for (int round = 0; round < DIRECT_COLUMNS; round++)
        {
            directDot<T, A, B, E, S>(wg.z, wg.y * DIRECT_ROWS, col + round * threads, a, b, epilogue, out, M, N, K);
        }
    }
    else
    {
        matmulTile<T, VEC, TILE_M, A, B, E, S>(li, wg, wg.z, smemA, smemB, a, b, epilogue, out, M, N, K);
    }
}

[shader("compute")]
[numthreads(WG_x, WG_y, 1)]
void conv_main<T : ITorchVulkan, let VEC : int, let TILE_M : int>(
    uint3 li : SV_GroupThreadID,
    uint3 wg : SV_GroupID,
    uniform ConvPushConstants pcs)
{
    typealias Acc = T.Accum;

    static groupshared Acc smemA[TILE_M * (TILE_K + 1)];
    static groupshared vector<Acc, VEC> smemB[TILE_K * (TILE_N / VEC)];

    uint lane = li.y * WG_x + li.x;
    if (lane < PARAMS) params[lane] = ((uint*)(*(Metadata*)pcs.metadata).params)[lane];
    GroupMemoryBarrierWithGroupSync();
    Geometry g = Geometry();
    uint taps = g.taps.d;

    if (MODE == 0)
    {
        runGemm<Acc, VEC, TILE_M, WeightRows<T>, UnfoldedColumns<T>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB,
            WeightRows<T>(pcs.weight, g), UnfoldedColumns<T>(pcs.x, g), BiasEpilogue<T>(pcs.bias, g.cy),
            GroupedStore<T>(pcs.y, g.groups, g.cy, g.outputs),
            g.cy, pcs.n * g.outputs.d, g.cx * taps);
    }
    else if (MODE == 1)
    {
        runGemm<Acc, VEC, TILE_M, WeightColumns<T>, GatheredOutput<T>, BiasEpilogue<T>, GroupedStore<T>>(
            li, wg, smemA, smemB,
            WeightColumns<T>(pcs.weight, g), GatheredOutput<T>(pcs.y, g), BiasEpilogue<T>(pcs.bias, g.cx),
            GroupedStore<T>(pcs.x, g.groups, g.cx, g.inputs),
            g.cx, pcs.n * g.inputs.d, g.cy * taps);
    }
    else
    {
        runGemm<Acc, VEC, TILE_M, OutputRows<T>, UnfoldedRows<T>, BiasEpilogue<T>, BlockStore<T>>(
            li, wg, smemA, smemB,
            OutputRows<T>(pcs.y, g, pcs.n, pcs.chunk), UnfoldedRows<T>(pcs.x, g, pcs.n, pcs.chunk), BiasEpilogue<T>(pcs.bias, g.cy),
            BlockStore<T>(pcs.weight, g.cy, g.cx * taps),
            g.cy, g.cx * taps, pcs.chunk);
    }
}

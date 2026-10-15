module fft;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint INVERSE = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct FftPushConstants
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t aux;
    public uint64_t count;
    public uint n;
    public uint m;
    public uint half_span;
    public uint padding;
}

static const float PI = 3.14159265358979323846f;

// the angle is reduced first, so large indices keep their precision in float32
float2 unitRoot(float turns)
{
    float angle = 2.0f * PI * turns;
    return float2(cos(angle), sin(angle));
}

float2 complexMul(float2 a, float2 b) { return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

float2 load<T : __BuiltinFloatingPointType>(uint64_t address, uint64_t index)
{
    T* data = (T*)address;
    return float2(data[index * 2].toFloat(), data[index * 2 + 1].toFloat());
}

void store<T : __BuiltinFloatingPointType>(uint64_t address, uint64_t index, float2 v)
{
    T* data = (T*)address;
    data[index * 2] = T(v.x);
    data[index * 2 + 1] = T(v.y);
}

// j^2 is reduced modulo 2n, so float32 keeps its precision
float2 chirp(uint j, uint n, bool conjugate)
{
    uint64_t square = (uint64_t(j) * uint64_t(j)) % (uint64_t(2) * n);
    float turns = float(square) / float(2 * n);
    bool negative = (INVERSE == 0) != conjugate;
    return unitRoot(negative ? -turns : turns);
}

void fftKernel<T : __BuiltinFloatingPointType>(uint64_t idx, FftPushConstants pcs)
{
    if (idx >= pcs.count) return;

    if (MODE == 0)
    {
        // radix-2 Stockham stage
        uint half = pcs.n / 2;
        uint64_t row = idx / half;
        uint j = uint(idx % half);
        uint span = pcs.half_span;
        uint k = j % span;
        uint64_t base = row * pcs.n;
        float2 v0 = load<T>(pcs.input, base + j);
        float2 v1 = load<T>(pcs.input, base + j + half);
        float turns = float(k) / float(2 * span);
        v1 = complexMul(v1, unitRoot((INVERSE != 0) ? turns : -turns));
        uint out_index = (j / span) * span * 2 + k;
        store<T>(pcs.output, base + out_index, v0 + v1);
        store<T>(pcs.output, base + out_index + span, v0 - v1);
        return;
    }

    if (MODE == 1)
    {
        uint64_t row = idx / pcs.m;
        uint j = uint(idx % pcs.m);
        float2 v = float2(0.0f, 0.0f);
        if (j < pcs.n) v = complexMul(load<T>(pcs.input, row * pcs.n + j), chirp(j, pcs.n, false));
        store<T>(pcs.output, idx, v);
        return;
    }

    if (MODE == 2)
    {
        uint j = uint(idx);
        uint distance = (j < pcs.n) ? j : ((pcs.m - j < pcs.n) ? pcs.m - j : pcs.n);
        float2 v = (distance < pcs.n) ? chirp(distance, pcs.n, true) : float2(0.0f, 0.0f);
        store<T>(pcs.output, idx, v);
        return;
    }

    if (MODE == 3)
    {
        uint j = uint(idx % pcs.m);
        store<T>(pcs.output, idx, complexMul(load<T>(pcs.input, idx), load<T>(pcs.aux, j)));
        return;
    }

    uint64_t row = idx / pcs.n;
    uint k = uint(idx % pcs.n);
    float2 c = load<T>(pcs.input, row * pcs.m + k);
    store<T>(pcs.output, idx, complexMul(c, chirp(k, pcs.n, false)) / float(pcs.m));
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void fft_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform FftPushConstants pcs)
{
    fftKernel<T>(uint64_t(gi.x), pcs);
}

module dropout;

__exported import core.philox;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct DropoutPushConstants
{
    public uint64_t input;
    public uint64_t output;
    public uint64_t mask;
    public uint64_t numel;
    public uint64_t seed;
    public uint64_t offset;
    public uint64_t keep_prob; // a T.Real's bits
    public uint64_t scale;     // a T.Real's bits
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void dropout_main<T : IRandomDtype, let VEC_SIZE : int>(
    uint3 gi : SV_DispatchThreadID,
    uniform DropoutPushConstants pcs)
{
    typealias R = T.Real;

    uint stream_id = flat_thread(gi, WG_x);
    uint first = stream_id * VEC_SIZE;
    if (first >= pcs.numel) return;

    var stream = RandomStream<R>(pcs.seed, pcs.offset, stream_id);
    R keepProb = R.fromBits(pcs.keep_prob);
    R scale = R.fromBits(pcs.scale);

    T* input = (T*)pcs.input;
    T* output = (T*)pcs.output;
    uint8_t* mask = (uint8_t*)pcs.mask;

    for (int k = 0; k < VEC_SIZE; k++)
    {
        uint element = first + k;
        if (element >= pcs.numel) return;

        bool kept = stream.uniform() < keepProb;
        // multiplying by zero rather than writing it keeps torch's nan for a dropped inf
        R value = castValue<T, R>(input[element]) * (kept ? scale : R(0));
        output[element] = castValue<R, T>(value);
        mask[element] = kept ? uint8_t(1) : uint8_t(0);
    }
}

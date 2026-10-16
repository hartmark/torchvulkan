module multinomial;

__exported import core.philox;
import core.indexing;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct MultinomialPushConstants
{
    public uint64_t cdf;    // [distributions, categories] running sums of the probabilities
    public uint64_t output; // [distributions, samples] int64 category indices
    public uint64_t numel;
    public uint64_t samples;
    public uint64_t categories;
    public uint64_t seed;
    public uint64_t offset;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void multinomial_main<R : __BuiltinFloatingPointType & IRandomReal>(
    uint3 gi : SV_DispatchThreadID,
    uniform MultinomialPushConstants pcs)
{
    uint index = flat_thread(gi, WG_x);
    if (index >= pcs.numel) return;

    R* cdf = (R*)pcs.cdf + (index / pcs.samples) * pcs.categories;
    var stream = RandomStream<R>(pcs.seed, pcs.offset, index);
    R target = stream.uniform() * cdf[pcs.categories - 1];

    // a zero-probability category repeats its predecessor's sum, so it is never the first to exceed
    uint64_t low = 0;
    uint64_t high = pcs.categories - 1;
    while (low < high)
    {
        uint64_t middle = (low + high) / 2;
        if (cdf[middle] > target) high = middle;
        else low = middle + 1;
    }

    ((int64_t*)pcs.output)[index] = int64_t(low);
}

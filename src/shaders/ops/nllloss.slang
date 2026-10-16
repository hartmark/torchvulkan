module nllloss;

__exported import core.dtype;
import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct NllLossPushConstants
{
    public uint64_t tensor_in;
    public uint64_t target;
    public uint64_t tensor_out;
    public int64_t ignore_index;
    public uint N;
    public uint C;
}

// a target outside [0, C) contributes nothing, instead of reading out of bounds
bool targetInRange(int64_t t, uint C)
{
    return t >= 0 && t < int64_t(C);
}

public interface INllLossMode
{
    static void apply<T : ITorchVulkan>(uint tid, T* tensor_in, int64_t* target, T* tensor_out, NllLossPushConstants pcs);
}

public struct ForwardGatherOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(uint tid, T* tensor_in, int64_t* target, T* tensor_out, NllLossPushConstants pcs)
    {
        if (tid >= pcs.N) return;

        int64_t t = target[tid];
        if (t == pcs.ignore_index || !targetInRange(t, pcs.C)) { tensor_out[tid] = T.ext_zero(); return; }
        tensor_out[tid] = T.ext_neg(tensor_in[uint64_t(tid) * pcs.C + uint64_t(t)]);
    }
}

public struct BackwardScatterOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(uint tid, T* tensor_in, int64_t* target, T* tensor_out, NllLossPushConstants pcs)
    {
        if (tid >= pcs.N * pcs.C) return;

        uint i = tid / pcs.C;
        uint c = tid % pcs.C;
        int64_t t = target[i];
        tensor_out[tid] = (t != pcs.ignore_index && int64_t(c) == t) ? T.ext_neg(tensor_in[i]) : T.ext_zero();
    }
}

public struct WeightGatherOp : INllLossMode
{
    public static void apply<T : ITorchVulkan>(uint tid, T* tensor_in, int64_t* target, T* tensor_out, NllLossPushConstants pcs)
    {
        if (tid >= pcs.N) return;

        int64_t t = target[tid];
        if (t == pcs.ignore_index || !targetInRange(t, pcs.C)) { tensor_out[tid] = T.ext_zero(); return; }
        tensor_out[tid] = tensor_in[uint64_t(t)];
    }
}

public void dispatchNllLoss<T : ITorchVulkan>(uint3 gi, NllLossPushConstants pcs)
{
    T* tensor_in = (T*)pcs.tensor_in;
    int64_t* target = (int64_t*)pcs.target;
    T* tensor_out = (T*)pcs.tensor_out;

    switch (MODE)
    {
    case 0: ForwardGatherOp.apply<T>(flat_thread(gi, WG_x), tensor_in, target, tensor_out, pcs); break;
    case 1: BackwardScatterOp.apply<T>(flat_thread(gi, WG_x), tensor_in, target, tensor_out, pcs); break;
    case 2: WeightGatherOp.apply<T>(flat_thread(gi, WG_x), tensor_in, target, tensor_out, pcs); break;
    default: break;
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void nllloss_main<T : ITorchVulkan>(
    uint3 gi : SV_DispatchThreadID,
    uniform NllLossPushConstants pcs)
{
    dispatchNllLoss<T>(gi, pcs);
}

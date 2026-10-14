module index_gather;

import core.indexing;

[[vk::constant_id(0)]] public const uint NDIM = 0;
[[vk::constant_id(1)]] public const uint NUM_INDICES = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;
[[vk::constant_id(3)]] public const uint WRAP_NEGATIVE = 0;

static const uint INDEX_MAX_TENSORS = 4;

public struct Metadata
{
    public uint64_t sizes;
    public uint64_t strides_out;
    public uint64_t strides_src;
    public uint64_t strides_index[INDEX_MAX_TENSORS];
    public uint64_t index_bounds;
}

public struct IndexGatherPushConstants
{
    public uint64_t tensor_out;
    public uint64_t tensor_src;
    public uint64_t indices[INDEX_MAX_TENSORS];
    public uint64_t metadata;
    public uint64_t numel;
}

public void indexGatherKernel<T>(uint3 gi, IndexGatherPushConstants pcs)
{
    uint idx = gi.x;
    if (idx >= pcs.numel) return;

    T* tensor_out = (T*)pcs.tensor_out;
    T* tensor_src = (T*)pcs.tensor_src;

    Metadata md = *(Metadata*)pcs.metadata;
    uint* strides_out = (uint*)md.strides_out;
    uint* strides_src = (uint*)md.strides_src;
    uint* index_bounds = (uint*)md.index_bounds;

    uint idx_out = 0;
    uint idx_src = 0;
    uint idx_index[INDEX_MAX_TENSORS];
    [ForceUnroll] for (int k = 0; k < INDEX_MAX_TENSORS; k++) idx_index[k] = 0;

    var walker = IndexWalker(md.sizes, NDIM, idx);
    for (uint i = 0; i < NDIM; i++)
    {
        uint coord = walker.next(i);
        idx_out += coord * strides_out[i];
        idx_src += coord * strides_src[i];
        [ForceUnroll] for (int k = 0; k < INDEX_MAX_TENSORS; k++)
        {
            if (k < NUM_INDICES) idx_index[k] += coord * ((uint*)md.strides_index[k])[i];
        }
    }

    int64_t offset = int64_t(idx_src);
    [ForceUnroll] for (int k = 0; k < INDEX_MAX_TENSORS; k++)
    {
        if (k >= NUM_INDICES) continue;

        int64_t size = int64_t(index_bounds[k]);
        int64_t index = ((int64_t*)pcs.indices[k])[idx_index[k]];
        if (WRAP_NEGATIVE == 1 && index < 0) index += size;
        if (index < 0 || index >= size) return;
        offset += index * int64_t(index_bounds[NUM_INDICES + k]);
    }

    tensor_out[idx_out] = tensor_src[offset];
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void index_gather_main<T>(
    uint3 gi : SV_DispatchThreadID,
    uniform IndexGatherPushConstants pcs)
{
    indexGatherKernel<T>(gi, pcs);
}

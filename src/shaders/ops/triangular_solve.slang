module triangular_solve;

import core.indexing;

[[vk::constant_id(0)]] public const uint UPPER = 0;
[[vk::constant_id(1)]] public const uint UNIT_DIAGONAL = 0;
[[vk::constant_id(2)]] public const uint WG_x = 0;

public struct TriangularSolvePushConstants
{
    public uint64_t a;
    public uint64_t x;
    public uint64_t count;
    public uint n;
    public uint k;
}

// in dtrsm's order: ascending for lower A, descending for upper
void triangularSolveKernel<T : __BuiltinFloatingPointType>(uint64_t idx, TriangularSolvePushConstants pcs)
{
    if (idx >= pcs.count) return;
    int n = int(pcs.n);
    int k = int(pcs.k);
    int64_t batch = int64_t(idx) / k;
    int column = int(int64_t(idx) % k);
    T* a = (T*)pcs.a + batch * n * n;
    T* x = (T*)pcs.x + batch * n * k + column;

    if (UPPER == 0)
    {
        for (int i = 0; i < n; i++)
        {
            T s = x[int64_t(i) * k];
            for (int j = 0; j < i; j++) s -= x[int64_t(j) * k] * a[i * n + j];
            x[int64_t(i) * k] = (UNIT_DIAGONAL != 0) ? s : s / a[i * n + i];
        }
    }
    else
    {
        for (int i = n - 1; i >= 0; i--)
        {
            T s = x[int64_t(i) * k];
            for (int j = n - 1; j > i; j--) s -= x[int64_t(j) * k] * a[i * n + j];
            x[int64_t(i) * k] = (UNIT_DIAGONAL != 0) ? s : s / a[i * n + i];
        }
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void triangular_solve_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform TriangularSolvePushConstants pcs)
{
    triangularSolveKernel<T>(uint64_t(flat_thread(gi, WG_x)), pcs);
}

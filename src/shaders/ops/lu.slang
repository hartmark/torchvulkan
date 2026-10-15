module lu;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct LuPushConstants
{
    public uint64_t a;
    public uint64_t pivots;
    public uint m;
    public uint n;
}

// LAPACK's dgetf2
void luKernel<T : __BuiltinFloatingPointType>(uint b, uint lane, LuPushConstants pcs)
{
    static groupshared int pivot_row;
    int m = int(pcs.m);
    int n = int(pcs.n);
    int steps = min(m, n);
    T* a = (T*)pcs.a + int64_t(b) * m * n;
    int* pivots = (int*)pcs.pivots + int64_t(b) * steps;

    for (int j = 0; j < steps; j++)
    {
        if (lane == 0)
        {
            int best = j;
            T best_value = abs(a[j * n + j]);
            for (int i = j + 1; i < m; i++)
            {
                T v = abs(a[i * n + j]);
                if (v > best_value || (isnan(v) && !isnan(best_value)))
                {
                    best = i;
                    best_value = v;
                }
            }
            pivot_row = best;
            pivots[j] = best + 1;
        }
        AllMemoryBarrierWithGroupSync();

        int p = pivot_row;
        bool nonzero = a[p * n + j] != T(0);
        bool swap = nonzero && p != j;
        for (int base = 0; base < n; base += int(WG_x))
        {
            int c = base + int(lane);
            T top = T(0);
            T bottom = T(0);
            if (swap && c < n)
            {
                top = a[j * n + c];
                bottom = a[p * n + c];
            }
            AllMemoryBarrierWithGroupSync();
            if (swap && c < n)
            {
                a[j * n + c] = bottom;
                a[p * n + c] = top;
            }
            AllMemoryBarrierWithGroupSync();
        }

        if (nonzero)
        {
            T reciprocal = T(1) / a[j * n + j];
            for (int i = j + 1 + int(lane); i < m; i += int(WG_x)) a[i * n + j] *= reciprocal;
        }
        AllMemoryBarrierWithGroupSync();

        int rows = m - j - 1;
        int cols = n - j - 1;
        for (int idx = int(lane); idx < rows * cols; idx += int(WG_x))
        {
            int i = j + 1 + idx / cols;
            int c = j + 1 + idx % cols;
            T update = -a[j * n + c];
            a[i * n + c] += a[i * n + j] * update;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void lu_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uniform LuPushConstants pcs)
{
    luKernel<T>(group.x, thread.x, pcs);
}

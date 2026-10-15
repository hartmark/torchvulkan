module cholesky;

[[vk::constant_id(0)]] public const uint WG_x = 0;

public struct CholeskyPushConstants
{
    public uint64_t a;
    public uint64_t info;
    public uint n;
}

// LAPACK's dpotf2
void choleskyKernel<T : __BuiltinFloatingPointType>(uint b, uint lane, CholeskyPushConstants pcs)
{
    static groupshared int failed;
    int n = int(pcs.n);
    T* a = (T*)pcs.a + int64_t(b) * n * n;
    if (lane == 0) failed = 0;
    AllMemoryBarrierWithGroupSync();

    for (int j = 0; j < n; j++)
    {
        if (lane == 0)
        {
            T s = a[j * n + j];
            for (int k = 0; k < j; k++) s -= a[j * n + k] * a[j * n + k];
            if (!(s > T(0)))
            {
                failed = j + 1;
                a[j * n + j] = s;
            }
            else
            {
                a[j * n + j] = sqrt(s);
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (failed != 0) break;

        T reciprocal = T(1) / a[j * n + j];
        for (int i = j + 1 + int(lane); i < n; i += int(WG_x))
        {
            T s = a[i * n + j];
            for (int k = 0; k < j; k++) s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s * reciprocal;
        }
        AllMemoryBarrierWithGroupSync();
    }

    for (int idx = int(lane); idx < n * n; idx += int(WG_x))
    {
        if (idx % n > idx / n) a[idx] = T(0);
    }
    if (lane == 0) ((int*)pcs.info)[b] = failed;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void cholesky_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uniform CholeskyPushConstants pcs)
{
    choleskyKernel<T>(group.x, thread.x, pcs);
}

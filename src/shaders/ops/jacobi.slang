module jacobi;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

static const int MAX_SWEEPS = 60;

public struct JacobiPushConstants
{
    public uint64_t a;
    public uint64_t v;
    public uint64_t rotations;
    public uint m;
    public uint n;
    public uint pairs;
}

T epsilon<T : __BuiltinFloatingPointType>() { return (sizeof(T) == 8) ? T(2.220446049250313e-16) : T(1.1920928955078125e-07); }

void roundPair(int round, int i, int players, out int p, out int q)
{
    int left = (i == 0) ? 0 : 1 + (round + i - 1) % (players - 1);
    int j = players - 1 - i;
    int right = 1 + (round + j - 1) % (players - 1);
    p = min(left, right);
    q = max(left, right);
}

bool computeRotation<T : __BuiltinFloatingPointType>(T* a, int m, int n, int p, int q, out T c, out T s)
{
    c = T(1);
    s = T(0);
    T app;
    T aqq;
    T apq;
    if (MODE == 0)
    {
        app = T(0);
        aqq = T(0);
        apq = T(0);
        for (int i = 0; i < m; i++)
        {
            T x = a[i * n + p];
            T y = a[i * n + q];
            app += x * x;
            aqq += y * y;
            apq += x * y;
        }
        if (apq == T(0) || abs(apq) <= epsilon<T>() * sqrt(app * aqq)) return false;
    }
    else
    {
        app = a[p * n + p];
        aqq = a[q * n + q];
        apq = a[p * n + q];
        if (apq == T(0) || abs(apq) <= epsilon<T>() * sqrt(abs(app) * abs(aqq)) * T(0.5)) return false;
    }
    T zeta = (aqq - app) / (T(2) * apq);
    T t = ((zeta >= T(0)) ? T(1) : T(-1)) / (abs(zeta) + sqrt(T(1) + zeta * zeta));
    c = T(1) / sqrt(T(1) + t * t);
    s = c * t;
    return true;
}

void applyRotations<T : __BuiltinFloatingPointType>(T* data, T* rotations, uint lane, int pairs, int round, int players, int length, int line_stride,
    int element_stride, int n)
{
    int items = pairs * length;
    for (int base = 0; base < items; base += int(WG_x))
    {
        int idx = base + int(lane);
        T x = T(0);
        T y = T(0);
        T c = T(1);
        T s = T(0);
        int64_t px = 0;
        int64_t qx = 0;
        bool active = false;
        if (idx < items)
        {
            int pair = idx / length;
            int e = idx % length;
            int p;
            int q;
            roundPair(round, pair, players, p, q);
            c = rotations[pair * 2];
            s = rotations[pair * 2 + 1];
            active = q < n && s != T(0);
            px = int64_t(p) * line_stride + int64_t(e) * element_stride;
            qx = int64_t(q) * line_stride + int64_t(e) * element_stride;
            if (active)
            {
                x = data[px];
                y = data[qx];
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (active)
        {
            data[px] = c * x - s * y;
            data[qx] = s * x + c * y;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

void jacobiKernel<T : __BuiltinFloatingPointType>(uint b, uint lane, JacobiPushConstants pcs)
{
    static groupshared int rotated;
    int m = int(pcs.m);
    int n = int(pcs.n);
    int pairs = int(pcs.pairs);
    int players = pairs * 2;
    T* a = (T*)pcs.a + int64_t(b) * m * n;
    T* v = (T*)pcs.v + int64_t(b) * n * n;
    T* rotations = (T*)pcs.rotations + int64_t(b) * pairs * 2;

    for (int idx = int(lane); idx < n * n; idx += int(WG_x)) v[idx] = (idx / n == idx % n) ? T(1) : T(0);
    AllMemoryBarrierWithGroupSync();

    for (int sweep = 0; sweep < MAX_SWEEPS && players > 1; sweep++)
    {
        if (lane == 0) rotated = 0;
        AllMemoryBarrierWithGroupSync();

        for (int round = 0; round < players - 1; round++)
        {
            for (int pair = int(lane); pair < pairs; pair += int(WG_x))
            {
                int p;
                int q;
                roundPair(round, pair, players, p, q);
                T c = T(1);
                T s = T(0);
                if (q < n && computeRotation<T>(a, m, n, p, q, c, s)) rotated = 1;
                rotations[pair * 2] = c;
                rotations[pair * 2 + 1] = s;
            }
            AllMemoryBarrierWithGroupSync();

            if (MODE == 0)
            {
                applyRotations<T>(a, rotations, lane, pairs, round, players, m, 1, n, n);
            }
            else
            {
                applyRotations<T>(a, rotations, lane, pairs, round, players, n, n, 1, n);
                applyRotations<T>(a, rotations, lane, pairs, round, players, n, 1, n, n);
            }
            applyRotations<T>(v, rotations, lane, pairs, round, players, n, 1, n, n);
        }
        AllMemoryBarrierWithGroupSync();
        if (rotated == 0) break;
        AllMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void jacobi_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uniform JacobiPushConstants pcs)
{
    jacobiKernel<T>(group.x, thread.x, pcs);
}

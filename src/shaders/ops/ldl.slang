module ldl;

import core.indexing;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct LdlPushConstants
{
    public uint64_t a;
    public uint64_t pivots;
    public uint64_t info;
    public uint64_t b;
    public uint64_t scratch;
    public uint64_t count;
    public uint n;
    public uint k;
}

// LAPACK's A(i, j), 1-based, on a row-major matrix
struct Matrix<T : __BuiltinFloatingPointType>
{
    T* data;
    int n;
    int stride;

    T get(int i, int j) { return data[int64_t(i - 1) * stride + (j - 1)]; }
    [mutating] void set(int i, int j, T v) { data[int64_t(i - 1) * stride + (j - 1)] = v; }

    // staged through scratch: MoltenVK's translation can re-read a loaded value at its use, after the swap
    [mutating] void swap(int i1, int j1, int i2, int j2, T* scratch)
    {
        scratch[0] = get(i1, j1);
        scratch[1] = get(i2, j2);
        set(i1, j1, scratch[1]);
        set(i2, j2, scratch[0]);
    }
}

// LAPACK's idamax, 1-based
int idamax<T : __BuiltinFloatingPointType>(Matrix<T> a, int count, int i, int j, bool along_row)
{
    int best = 1;
    T best_value = T(-1);
    for (int t = 0; t < count; t++)
    {
        T v = along_row ? abs(a.get(i, j + t)) : abs(a.get(i + t, j));
        if (v > best_value || (isnan(v) && !isnan(best_value)))
        {
            best = t + 1;
            best_value = v;
        }
    }
    return best;
}

// LAPACK's dsytf2 with UPLO = 'L' (Bunch-Kaufman)
void factor<T : __BuiltinFloatingPointType>(int64_t batch, LdlPushConstants pcs)
{
    int n = int(pcs.n);
    var a = Matrix<T>((T*)pcs.a + batch * n * n, n, n);
    int* ipiv = (int*)pcs.pivots + batch * n;
    T* scratch = (T*)pcs.scratch + batch * 2;
    T alpha = (T(1) + sqrt(T(17))) / T(8);
    int info = 0;

    int k = 1;
    while (k <= n)
    {
        int kstep = 1;
        int kp = k;
        T absakk = abs(a.get(k, k));
        int imax = k;
        T colmax = T(0);
        if (k < n)
        {
            imax = k + idamax<T>(a, n - k, k + 1, k, false);
            colmax = abs(a.get(imax, k));
        }
        if (max(absakk, colmax) == T(0) || isnan(absakk))
        {
            if (info == 0) info = k;
            kp = k;
        }
        else
        {
            if (absakk >= alpha * colmax)
            {
                kp = k;
            }
            else
            {
                int jmax = k - 1 + idamax<T>(a, imax - k, imax, k, true);
                T rowmax = abs(a.get(imax, jmax));
                if (imax < n)
                {
                    jmax = imax + idamax<T>(a, n - imax, imax + 1, imax, false);
                    rowmax = max(rowmax, abs(a.get(jmax, imax)));
                }
                if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
                else if (abs(a.get(imax, imax)) >= alpha * rowmax) kp = imax;
                else
                {
                    kp = imax;
                    kstep = 2;
                }
            }

            int kk = k + kstep - 1;
            if (kp != kk)
            {
                for (int i = kp + 1; i <= n; i++) a.swap(i, kk, i, kp, scratch);
                for (int t = 0; t < kp - kk - 1; t++) a.swap(kk + 1 + t, kk, kp, kk + 1 + t, scratch);
                a.swap(kk, kk, kp, kp, scratch);
                if (kstep == 2) a.swap(k + 1, k, kp, k, scratch);
            }

            if (kstep == 1)
            {
                if (k < n)
                {
                    T d11 = T(1) / a.get(k, k);
                    for (int j = k + 1; j <= n; j++)
                    {
                        T xj = a.get(j, k);
                        if (xj != T(0))
                        {
                            T temp = -d11 * xj;
                            for (int i = j; i <= n; i++) a.set(i, j, a.get(i, j) + a.get(i, k) * temp);
                        }
                    }
                    for (int i = k + 1; i <= n; i++) a.set(i, k, a.get(i, k) * d11);
                }
            }
            else
            {
                if (k < n - 1)
                {
                    T d21 = a.get(k + 1, k);
                    T d11 = a.get(k + 1, k + 1) / d21;
                    T d22 = a.get(k, k) / d21;
                    T t = T(1) / (d11 * d22 - T(1));
                    d21 = t / d21;
                    for (int j = k + 2; j <= n; j++)
                    {
                        T wk = d21 * (d11 * a.get(j, k) - a.get(j, k + 1));
                        T wkp1 = d21 * (d22 * a.get(j, k + 1) - a.get(j, k));
                        for (int i = j; i <= n; i++) a.set(i, j, a.get(i, j) - a.get(i, k) * wk - a.get(i, k + 1) * wkp1);
                        a.set(j, k, wk);
                        a.set(j, k + 1, wkp1);
                    }
                }
            }
        }
        if (kstep == 1)
        {
            ipiv[k - 1] = kp;
        }
        else
        {
            ipiv[k - 1] = -kp;
            ipiv[k] = -kp;
        }
        k += kstep;
    }
    ((int*)pcs.info)[batch] = info;
}

// LAPACK's dsytrs with UPLO = 'L'
void solve<T : __BuiltinFloatingPointType>(int64_t batch, int column, LdlPushConstants pcs)
{
    int n = int(pcs.n);
    int k_rhs = int(pcs.k);
    var a = Matrix<T>((T*)pcs.a + batch * n * n, n, n);
    var b = Matrix<T>((T*)pcs.b + batch * n * k_rhs + column, n, k_rhs);
    int* ipiv = (int*)pcs.pivots + batch * n;
    T* scratch = (T*)pcs.scratch + (batch * k_rhs + column) * 2;

    int k = 1;
    while (k <= n)
    {
        if (ipiv[k - 1] > 0)
        {
            int kp = ipiv[k - 1];
            if (kp != k) b.swap(k, 1, kp, 1, scratch);
            for (int i = k + 1; i <= n; i++) b.set(i, 1, b.get(i, 1) - a.get(i, k) * b.get(k, 1));
            b.set(k, 1, b.get(k, 1) * (T(1) / a.get(k, k)));
            k += 1;
        }
        else
        {
            int kp = -ipiv[k - 1];
            if (kp != k + 1) b.swap(k + 1, 1, kp, 1, scratch);
            if (k < n - 1)
            {
                for (int i = k + 2; i <= n; i++) b.set(i, 1, b.get(i, 1) - a.get(i, k) * b.get(k, 1));
                for (int i = k + 2; i <= n; i++) b.set(i, 1, b.get(i, 1) - a.get(i, k + 1) * b.get(k + 1, 1));
            }
            T akm1k = a.get(k + 1, k);
            T akm1 = a.get(k, k) / akm1k;
            T ak = a.get(k + 1, k + 1) / akm1k;
            T denom = akm1 * ak - T(1);
            T bkm1 = b.get(k, 1) / akm1k;
            T bk = b.get(k + 1, 1) / akm1k;
            b.set(k, 1, (ak * bkm1 - bk) / denom);
            b.set(k + 1, 1, (akm1 * bk - bkm1) / denom);
            k += 2;
        }
    }

    k = n;
    while (k >= 1)
    {
        if (ipiv[k - 1] > 0)
        {
            if (k < n)
            {
                T s = T(0);
                for (int i = k + 1; i <= n; i++) s += b.get(i, 1) * a.get(i, k);
                b.set(k, 1, b.get(k, 1) - s);
            }
            int kp = ipiv[k - 1];
            if (kp != k) b.swap(k, 1, kp, 1, scratch);
            k -= 1;
        }
        else
        {
            if (k < n)
            {
                T s = T(0);
                for (int i = k + 1; i <= n; i++) s += b.get(i, 1) * a.get(i, k);
                b.set(k, 1, b.get(k, 1) - s);
                T s2 = T(0);
                for (int i = k + 1; i <= n; i++) s2 += b.get(i, 1) * a.get(i, k - 1);
                b.set(k - 1, 1, b.get(k - 1, 1) - s2);
            }
            int kp = -ipiv[k - 1];
            if (kp != k) b.swap(k, 1, kp, 1, scratch);
            k -= 2;
        }
    }
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void ldl_main<T : __BuiltinFloatingPointType>(
    uint3 gi : SV_DispatchThreadID,
    uniform LdlPushConstants pcs)
{
    uint64_t idx = uint64_t(flat_thread(gi, WG_x));
    if (idx >= pcs.count) return;
    if (MODE == 0) factor<T>(int64_t(idx), pcs);
    else solve<T>(int64_t(idx) / int64_t(pcs.k), int(int64_t(idx) % int64_t(pcs.k)), pcs);
}

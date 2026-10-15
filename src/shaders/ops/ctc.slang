module ctc;

[[vk::constant_id(0)]] public const uint MODE = 0;
[[vk::constant_id(1)]] public const uint WG_x = 0;

public struct CtcPushConstants
{
    public uint64_t log_probs;
    public uint64_t targets;
    public uint64_t target_offsets;
    public uint64_t input_lengths;
    public uint64_t target_lengths;
    public uint64_t log_alpha;
    public uint64_t log_beta;
    public uint64_t nll;
    public uint64_t grad_out;
    public uint64_t grad;
    public uint time;
    public uint batch;
    public uint labels;
    public uint states;
    public uint blank;
    public uint zero_infinity;
}

int64_t targetPrime(CtcPushConstants pcs, int64_t b, int s)
{
    if (s % 2 == 0) return int64_t(pcs.blank);
    int64_t offset = ((int64_t*)pcs.target_offsets)[b];
    return ((int64_t*)pcs.targets)[offset + int64_t(s / 2)];
}

T negInf<T : __BuiltinFloatingPointType>() { return T(-1) / T(0); }

T expOf<T : __BuiltinFloatingPointType>(T x) { if (sizeof(T) == 8) return T(exp(x.toFloat())); return exp(x); }
T logOf<T : __BuiltinFloatingPointType>(T x) { if (sizeof(T) == 8) return T(log(x.toFloat())); return log(x); }

T logSumExp3<T : __BuiltinFloatingPointType>(T a, T b, T c)
{
    T m = max(max(a, b), c);
    if (m == negInf<T>()) m = T(0);
    return logOf<T>(expOf<T>(a - m) + expOf<T>(b - m) + expOf<T>(c - m)) + m;
}

T logProb<T : __BuiltinFloatingPointType>(CtcPushConstants pcs, int t, int64_t b, int64_t label)
{
    T* log_probs = (T*)pcs.log_probs;
    return log_probs[(int64_t(t) * pcs.batch + b) * pcs.labels + label];
}

void alphaKernel<T : __BuiltinFloatingPointType>(uint b_index, uint lane, CtcPushConstants pcs)
{
    int64_t b = int64_t(b_index);
    int input_length = int(((int64_t*)pcs.input_lengths)[b]);
    int target_length = int(((int64_t*)pcs.target_lengths)[b]);
    int states = int(pcs.states);
    int used = 2 * target_length + 1;
    T* alpha = (T*)pcs.log_alpha + b * int64_t(pcs.time) * states;

    for (int t = 0; t < int(pcs.time); t++)
    {
        for (int s = int(lane); s < states; s += int(WG_x))
        {
            T value = negInf<T>();
            if (t < input_length && s < used)
            {
                int64_t prime = targetPrime(pcs, b, s);
                if (t == 0)
                {
                    if (s <= 1) value = logProb<T>(pcs, 0, b, prime);
                }
                else
                {
                    T* previous = alpha + int64_t(t - 1) * states;
                    T a1 = previous[s];
                    T a2 = (s > 0) ? previous[s - 1] : negInf<T>();
                    T a3 = (s > 1 && targetPrime(pcs, b, s - 2) != prime) ? previous[s - 2] : negInf<T>();
                    value = logSumExp3<T>(a1, a2, a3) + logProb<T>(pcs, t, b, prime);
                }
            }
            alpha[int64_t(t) * states + s] = value;
        }
        AllMemoryBarrierWithGroupSync();
    }

    if (lane != 0) return;
    T* nll = (T*)pcs.nll;
    if (input_length == 0)
    {
        nll[b] = (target_length == 0) ? T(0) : -negInf<T>();
        return;
    }
    T* last = alpha + int64_t(input_length - 1) * states;
    if (target_length == 0)
    {
        nll[b] = -last[0];
        return;
    }
    T l1 = last[2 * target_length];
    T l2 = last[2 * target_length - 1];
    T m = max(l1, l2);
    if (m == negInf<T>()) m = T(0);
    nll[b] = -(logOf<T>(expOf<T>(l1 - m) + expOf<T>(l2 - m)) + m);
}

void betaKernel<T : __BuiltinFloatingPointType>(uint b_index, uint lane, CtcPushConstants pcs)
{
    int64_t b = int64_t(b_index);
    int input_length = int(((int64_t*)pcs.input_lengths)[b]);
    int target_length = int(((int64_t*)pcs.target_lengths)[b]);
    int states = int(pcs.states);
    int used = 2 * target_length + 1;
    T* beta = (T*)pcs.log_beta + b * int64_t(pcs.time) * states;

    for (int t = int(pcs.time) - 1; t >= 0; t--)
    {
        for (int s = int(lane); s < states; s += int(WG_x))
        {
            T value = negInf<T>();
            if (t < input_length && s < used)
            {
                int64_t prime = targetPrime(pcs, b, s);
                if (t == input_length - 1)
                {
                    if (s >= used - 2) value = logProb<T>(pcs, t, b, prime);
                }
                else
                {
                    T* next = beta + int64_t(t + 1) * states;
                    T b1 = next[s];
                    T b2 = (s < 2 * target_length) ? next[s + 1] : negInf<T>();
                    T b3 = (s < 2 * target_length - 1 && targetPrime(pcs, b, s + 2) != prime) ? next[s + 2] : negInf<T>();
                    value = logSumExp3<T>(b1, b2, b3) + logProb<T>(pcs, t, b, prime);
                }
            }
            beta[int64_t(t) * states + s] = value;
        }
        AllMemoryBarrierWithGroupSync();
    }
}

void gradKernel<T : __BuiltinFloatingPointType>(uint64_t idx, CtcPushConstants pcs)
{
    uint64_t count = uint64_t(pcs.time) * pcs.batch * pcs.labels;
    if (idx >= count) return;
    int64_t c = int64_t(idx % pcs.labels);
    int64_t b = int64_t((idx / pcs.labels) % pcs.batch);
    int t = int(idx / (uint64_t(pcs.labels) * pcs.batch));
    T* grad = (T*)pcs.grad;

    int input_length = int(((int64_t*)pcs.input_lengths)[b]);
    int target_length = int(((int64_t*)pcs.target_lengths)[b]);
    T nll = ((T*)pcs.nll)[b];
    bool zeroed = pcs.zero_infinity != 0 && nll == -negInf<T>();
    if (t >= input_length || zeroed)
    {
        grad[idx] = T(0);
        return;
    }

    int states = int(pcs.states);
    T* alpha = (T*)pcs.log_alpha + (b * int64_t(pcs.time) + t) * states;
    T* beta = (T*)pcs.log_beta + (b * int64_t(pcs.time) + t) * states;
    T lcab = negInf<T>();
    if (t == input_length - 1)
    {
        if (c == int64_t(pcs.blank)) lcab = alpha[2 * target_length] + beta[2 * target_length];
        if (target_length > 0 && targetPrime(pcs, b, 2 * target_length - 1) == c) lcab = alpha[2 * target_length - 1] + beta[2 * target_length - 1];
    }
    else
    {
        for (int s = 2 * target_length; s >= 0; s--)
        {
            if (targetPrime(pcs, b, s) != c) continue;
            T term = alpha[s] + beta[s];
            if (lcab == negInf<T>())
            {
                lcab = term;
            }
            else
            {
                T m = max(lcab, term);
                lcab = logOf<T>(expOf<T>(lcab - m) + expOf<T>(term - m)) + m;
            }
        }
    }
    T lp = logProb<T>(pcs, t, b, c);
    T gr = ((T*)pcs.grad_out)[b];
    grad[idx] = (expOf<T>(lp) - expOf<T>(lcab + nll - lp)) * gr;
}

[shader("compute")]
[numthreads(WG_x, 1, 1)]
void ctc_main<T : __BuiltinFloatingPointType>(
    uint3 group : SV_GroupID,
    uint3 thread : SV_GroupThreadID,
    uint3 gi : SV_DispatchThreadID,
    uniform CtcPushConstants pcs)
{
    if (MODE == 0) alphaKernel<T>(group.x, thread.x, pcs);
    else if (MODE == 1) betaKernel<T>(group.x, thread.x, pcs);
    else gradKernel<T>(uint64_t(gi.x), pcs);
}

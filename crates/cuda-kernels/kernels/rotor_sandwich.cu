__device__ void gp_pair_cl3(const float* a, const float* b, float* out) {
    #pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
        out[i] = 0.0f;
    }

    #pragma unroll
    for (unsigned int ai = 0; ai < 8u; ++ai) {
        float ca = a[ai];
        #pragma unroll
        for (unsigned int bi = 0; bi < 8u; ++bi) {
            float cb = b[bi];
            unsigned int result = ai ^ bi;
            unsigned int inversions = 0u;
            #pragma unroll
            for (unsigned int dim = 0; dim < 3u; ++dim) {
                if ((ai & (1u << dim)) != 0u) {
                    inversions += __popc(bi & ((1u << dim) - 1u));
                }
            }
            float sign = (inversions & 1u) == 0u ? 1.0f : -1.0f;
            out[result] += sign * ca * cb;
        }
    }
}

extern "C" __global__ void rotor_sandwich_cl3_kernel(
    const float* rotor,
    const float* mv,
    float* out,
    unsigned int batch,
    unsigned int rotor_is_batched
) {
    unsigned int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= batch) {
        return;
    }

    unsigned int base = sample * 8u;
    const float* r = rotor_is_batched != 0u ? rotor + base : rotor;
    const float* m = mv + base;

    float rev[8];
    float temp[8];
    float result[8];

    #pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
        rev[i] = r[i];
    }
    rev[3] = -rev[3];
    rev[5] = -rev[5];
    rev[6] = -rev[6];
    rev[7] = -rev[7];

    gp_pair_cl3(m, rev, temp);
    gp_pair_cl3(r, temp, result);

    #pragma unroll
    for (unsigned int i = 0; i < 8u; ++i) {
        out[base + i] = result[i];
    }
}

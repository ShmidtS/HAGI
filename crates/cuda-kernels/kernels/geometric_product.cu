extern "C" __global__ void geometric_product_cl3_kernel(
    const float* a,
    const float* b,
    float* out,
    unsigned int batch
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * 8u) {
        return;
    }

    unsigned int sample = idx / 8u;
    unsigned int blade = idx & 7u;
    unsigned int base = sample * 8u;
    float acc = 0.0f;

    #pragma unroll
    for (unsigned int ai = 0; ai < 8u; ++ai) {
        float ca = a[base + ai];
        #pragma unroll
        for (unsigned int bi = 0; bi < 8u; ++bi) {
            unsigned int result = ai ^ bi;
            if (result != blade) {
                continue;
            }

            unsigned int inversions = 0u;
            #pragma unroll
            for (unsigned int dim = 0; dim < 3u; ++dim) {
                if ((ai & (1u << dim)) != 0u) {
                    inversions += __popc(bi & ((1u << dim) - 1u));
                }
            }
            float sign = (inversions & 1u) == 0u ? 1.0f : -1.0f;
            acc += sign * ca * b[base + bi];
        }
    }

    out[idx] = acc;
}

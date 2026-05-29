__device__ void fused_gp_pair_cl3(const float* a, const float* b, float* out) {
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

extern "C" __global__ void fused_rotor_hrm_msa_kernel(
    const float* input,
    const float* geometric,
    const float* hrm_weights,
    const float* hrm_out,
    float* out,
    unsigned int total,
    unsigned int geometric_len,
    unsigned int hrm_weights_len,
    float route_scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float hrm = hrm_out[idx];
    float weight = hrm_weights[idx % hrm_weights_len];
    out[idx] = hrm + geometric[idx % geometric_len] * weight + route_scale;
}

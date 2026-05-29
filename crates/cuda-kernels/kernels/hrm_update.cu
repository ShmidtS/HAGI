extern "C" __global__ void hrm_update_kernel(
    const float* input,
    const float* weights,
    float* out,
    unsigned int total,
    unsigned int weights_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    out[idx] = input[idx] * weights[idx % weights_len];
}

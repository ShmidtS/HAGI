extern "C" __global__ void msa_route_score_kernel(
    const float* query,
    const float* keys,
    float* out,
    unsigned int slot_count,
    unsigned int key_dim
) {
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) {
        return;
    }

    const float* key = keys + slot * key_dim;
    float dot = 0.0f;
    for (unsigned int d = 0; d < key_dim; ++d) {
        dot += query[d] * key[d];
    }
    out[slot] = dot;
}

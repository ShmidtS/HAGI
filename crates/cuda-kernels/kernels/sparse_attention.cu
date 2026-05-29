extern "C" __global__ void sparse_attention_kernel(
    const float* query,
    const float* keys,
    const float* values,
    const float* weights,
    float* out,
    unsigned int bt_count,
    unsigned int hidden,
    unsigned int num_selected
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = bt_count * hidden;
    if (idx >= total) {
        return;
    }

    unsigned int bt = idx / hidden;
    unsigned int d = idx - bt * hidden;
    const float* q = query + bt * hidden;
    float scale = rsqrtf((float)hidden);
    float max_score = -3.402823466e+38f;

    for (unsigned int s = 0; s < num_selected; ++s) {
        const float* k = keys + s * hidden;
        float dot = 0.0f;
        for (unsigned int h = 0; h < hidden; ++h) {
            dot += q[h] * k[h];
        }
        float score = dot * scale;
        max_score = fmaxf(max_score, score);
    }

    float exp_sum = 0.0f;
    float acc = 0.0f;
    for (unsigned int s = 0; s < num_selected; ++s) {
        const float* k = keys + s * hidden;
        float dot = 0.0f;
        for (unsigned int h = 0; h < hidden; ++h) {
            dot += q[h] * k[h];
        }
        float e = expf(dot * scale - max_score);
        exp_sum += e;
        acc += weights[s] * e * values[s * hidden + d];
    }

    out[idx] = exp_sum > 0.0f ? acc / exp_sum : 0.0f;
}

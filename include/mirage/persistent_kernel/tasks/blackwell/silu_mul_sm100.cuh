 #pragma once
 #include "tasks/common/common_header.cuh"
 namespace kernel {
 
 template <typename T>
 __device__ __forceinline__ void silu_mul_sm100_task_impl(T const *w1x_w3x, // Shape: B, TopK, 2*Inter, compact
                                                    T *output_ptr, // Shape: B, TopK, Inter
                                                int batch_size,
                                                int topk_size,
                                                int inter_size) { 

    // Each thread processes output_token_per_thread tokens

    int out_size = batch_size * topk_size * inter_size;

    for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
        // Get w1x_w3x[b, topk, :]
        int batch_idx = i / (TopK_SIZE * INTER_SIZE);
        int topk_idx = (i / INTER_SIZE) % TopK_SIZE;
        int inter_idx = i % INTER_SIZE;
        int w1_index = batch_idx * TopK_SIZE * 2 * INTER_SIZE + topk_idx * 2 * INTER_SIZE + inter_idx;
        T w1x = w1x_w3x[w1_index];
        T w3x = w1x_w3x[w1_index + INTER_SIZE];

        // SiLU(w1x) * w3x
        T out = w1x / (1.0f + expf(-w1x)) *w3x;
        output_ptr[i] = out;
    }
 
 } // namespace kernel
 
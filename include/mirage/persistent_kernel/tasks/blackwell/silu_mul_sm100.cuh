 #pragma once
 #include "tasks/common/common_header.cuh"
 namespace kernel {
 
 template <typename T>
 __device__ __forceinline__ void silu_mul_sm100_task_impl(void const *w1x_w3x_ptr, // Shape: B, TopK, 2*Inter, compact
                                                    void *output_ptr, // Shape: B, TopK, Inter
                                                int batch_size,
                                                int topk_size,
                                                int inter_size) { 

    // Each thread processes output_token_per_thread tokens

    const T* w1x_w3x = static_cast<const T*>(w1x_w3x_ptr);
    T* output = static_cast<T*>(output_ptr);

    int out_size = batch_size * topk_size * inter_size;

    for (int i = threadIdx.x; i < out_size; i += blockDim.x) {
        // Get w1x_w3x[b, topk, :]
        int batch_idx = i / (topk_size * inter_size);
        int topk_idx = (i / inter_size) % topk_size;
        int inter_idx = i % inter_size;
        int w1_index = batch_idx * topk_size * 2 * inter_size + topk_idx * 2 * inter_size + inter_idx;
        T w1x = w1x_w3x[w1_index];
        T w3x = w1x_w3x[w1_index + inter_size];

        // SiLU(w1x) * w3x
        T out = w1x / (1.0f + expf(-w1x)) *w3x;
        output_ptr[i] = out;
    }
 
 } // namespace kernel
 
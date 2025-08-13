# flashinfer: adapted from sglang + vllm
# refer to sgl-kernel/tests/test_custom_allreduce.py from sglang

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py

import os
import time
import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

import flashinfer.comm as comm

import cuda.bindings.runtime as cudart

from cuda_utils import checkCudaErrors
from moe_utils import MoEParam
from dist_utils import ProcessGroupInfo, parallel_launch
from flashinfer.cute_dsl.kernel.sm100_intra_moe import IntraDispatchKernel
from kernel.sm100_grouped_gemm import run_grouped_gemm

def test_loop(dist_param: ProcessGroupInfo):

    '''
    Initialize per rank input tensors
    '''

    num_ranks = dist_param.world_size
    num_local_ranks = dist_param.world_local_size
    rank = dist_param.rank
    local_rank = dist_param.local_rank 
    node_rank = dist_param.node_rank
    device = dist_param.device
    group = dist.group.WORLD

    local_buffer_ptr_list, remote_buffer_ptr_list = comm.create_shared_all_to_all_buffer(64 * 4, group=group)

    for i in range(num_local_ranks):
        local_ptr = local_buffer_ptr_list[i]
        count_tensor = torch.ones((6, 6), dtype=torch.int32) * (local_rank * num_local_ranks + i)
        cudart.cudaMemcpy(
            local_ptr,
            count_tensor.data_ptr(),
            6 * 6 * 4,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

    dist.barrier(group=group)


    for i, remote_ptr in enumerate(remote_buffer_ptr_list):
        test_tensor = torch.zeros((6, 6), dtype=torch.int32)
        cudart.cudaMemcpy(
            test_tensor.data_ptr(),
            remote_ptr,
            6 * 6 * 4,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        print(f"rank{local_rank}: test_tensor from rank {i}: {test_tensor}")

    return

    
    dist.destroy_process_group()


if __name__ == "__main__":
    num_processes = 2
    parallel_launch(
        num_processes,
        test_loop,
    )

    '''
    MoE:
    Input (per rank):
    input_token_tensor: [num_tokens, hidden_dim]
    topk_scores: [num_token, num_experts] -> topk_weights: [num_token, num_activate_experts], topk_indices: [num_token, num_activate_experts]
    
    Output (per rank):
    output_token_tensor: [num_token, hidden_dim]
    '''
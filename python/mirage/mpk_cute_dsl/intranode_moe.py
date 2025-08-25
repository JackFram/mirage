import os
import time
import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import dtype as torch_dtype

import cuda.bindings.runtime as cudart
import comm 

from cuda_utils import checkCudaErrors
from moe_utils import MoEParam
from dist_utils import ProcessGroupInfo, parallel_launch
from kernel.sm100_intra_moe import IntraMoEKernel
from kernel.sm100_grouped_gemm import GroupedGemmKernel

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

    num_tokens_per_rank, hidden_dim, inter_dim, num_topk, num_experts = 64, 7168, 5120, 8, (32 // num_ranks) * num_ranks

    assert num_experts % num_ranks == 0, f"num_experts {num_experts} should be divisible by num_ranks {num_ranks}"
    num_local_experts = num_experts // num_ranks
    num_tokens = num_tokens_per_rank * num_ranks

    moe_param = MoEParam(
            num_experts=num_experts,
            num_topk=num_topk,
            hidden_dim=hidden_dim,
            inter_dim=inter_dim,
            num_tokens_per_rank=num_tokens_per_rank,
            in_dtype=cutlass.Float16, # BFloat16 has a bug in cute dsl when constructing the buffer
            out_dtype=cutlass.Float16,
        )

    # Initialize input tensors
    input_tensor = torch.randn((num_tokens_per_rank, hidden_dim), dtype=torch_dtype(moe_param.in_dtype), device='cuda')
    gate_scores = torch.randn((num_tokens_per_rank, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_indices = torch.topk(gate_scores, num_topk, dim=-1, largest=True, sorted=False)[1].to(torch.int32)
    topk_weights = torch.randn((num_tokens_per_rank, num_topk), dtype=torch.float32, device='cuda')
    local_token_send_count_per_expert = torch.zeros((num_experts, 1), dtype=torch.int32, device='cuda')
    rank_token_count = torch.zeros((1), dtype=torch.int32, device='cuda')
    # Meta Info tensors
    recv_num_token_per_rank = torch.zeros((num_local_experts * num_ranks, 1), dtype=torch.int32, device='cuda')
    max_token_per_rank = num_local_experts * num_tokens
    src_index = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_expert = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_offset = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_rank = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_token = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')

    # For grid Sync
    global_sync_semaphore = torch.zeros((2, 1), dtype=torch.int32, device='cuda')
    global_sync_semaphore_cute = from_dlpack(global_sync_semaphore, assumed_align=16)

    
    '''
    dispatch
    '''
    num_tokens_per_local_expert_recv = torch.empty(
        (num_local_experts, 1),
        dtype=torch.int32,
    ).fill_(0).cuda()

    dispatch_recv_token_tensor = torch.empty(
        (num_local_experts, num_tokens, 1, hidden_dim),
        dtype=torch_dtype(moe_param.in_dtype),
    ).fill_(0).cuda()

    '''
    ffn_grouped_gemm
    '''
    w1_tensor = torch.randn(hidden_dim, inter_dim, dtype=torch_dtype(moe_param.out_dtype), device='cuda')
    w2_tensor = torch.randn(inter_dim, hidden_dim, dtype=torch_dtype(moe_param.out_dtype), device='cuda')
    w3_tensor = torch.randn(hidden_dim, inter_dim, dtype=torch_dtype(moe_param.out_dtype), device='cuda')

    '''
    combine
    '''
    combine_send_token_tensor = torch.randn(
        (num_local_experts, num_tokens, 1, hidden_dim),
        dtype=torch_dtype(moe_param.out_dtype),
    ).cuda()

    output_tensor = torch.empty(
        (num_tokens_per_rank, hidden_dim),
        dtype=torch_dtype(moe_param.out_dtype),
    ).fill_(0).cuda()

    # Convert tensors to cute tensors
    input_tensor_cute = from_dlpack(input_tensor, assumed_align=16)
    topk_indices_cute = from_dlpack(topk_indices, assumed_align=16)
    num_tokens_per_local_expert_recv_cute = from_dlpack(num_tokens_per_local_expert_recv, assumed_align=16)
    dispatch_recv_token_tensor_cute = from_dlpack(dispatch_recv_token_tensor, assumed_align=16)
    combine_send_token_tensor_cute = from_dlpack(combine_send_token_tensor, assumed_align=16)
    output_tensor_cute = from_dlpack(output_tensor, assumed_align=16)
    local_token_send_count_per_expert_cute = from_dlpack(local_token_send_count_per_expert, assumed_align=16)
    rank_token_count_cute = from_dlpack(rank_token_count, assumed_align=16)

    recv_num_token_per_rank_cute = from_dlpack(recv_num_token_per_rank, assumed_align=16)
    src_index_cute = from_dlpack(src_index, assumed_align=16)
    src_expert_cute = from_dlpack(src_expert, assumed_align=16)
    src_offset_cute = from_dlpack(src_offset, assumed_align=16)
    src_rank_cute = from_dlpack(src_rank, assumed_align=16)
    src_token_cute = from_dlpack(src_token, assumed_align=16)

    w1_tensor_cute = from_dlpack(w1_tensor, assumed_align=16)
    w2_tensor_cute = from_dlpack(w2_tensor, assumed_align=16)
    w3_tensor_cute = from_dlpack(w3_tensor, assumed_align=16)

    intra_moe_kernel = IntraMoEKernel(
        moe_param=moe_param,
        dist_param=dist_param,
    )

    buffer_size_in_bytes = intra_moe_kernel.buffer_size_in_bytes

    # Construct buffer for dispatching and combining tokens
    local_buffer_ptr_list, remote_buffer_ptr_list = comm.create_shared_all_to_all_buffer(
        size_in_bytes=buffer_size_in_bytes,
    )

    # Construct buffer for recording token counts:
    count_buffer_ptr_list = comm.create_shared_buffer(
        size_in_bytes=4 * moe_param.num_tokens_per_rank,
    )

    def convert_list_to_tensor(l, dtype) -> tuple[torch.Tensor, cute.Tensor]:
        torch_tensor = torch.tensor(l, dtype=dtype).cuda()
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        return torch_tensor, cute_tensor
    
    local_buffer_ptr_tensor, local_buffer_ptr_cute = convert_list_to_tensor(local_buffer_ptr_list, torch.int64)
    remote_buffer_ptr_tensor, remote_buffer_ptr_cute = convert_list_to_tensor(remote_buffer_ptr_list, torch.int64)
    count_buffer_ptr_tensor, count_buffer_ptr_cute = convert_list_to_tensor(count_buffer_ptr_list, torch.int64)

    # Get current CUDA stream from PyTorch
    torch_stream = torch.cuda.current_stream()
    # Get the raw stream pointer as a CUstream
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    
    intra_moe_kernel_compiled = cute.compile(
        intra_moe_kernel,
        input_tensor_cute,
        topk_indices_cute,
        num_tokens_per_local_expert_recv_cute,
        local_token_send_count_per_expert_cute,
        rank_token_count_cute,
        dispatch_recv_token_tensor_cute,
        combine_send_token_tensor_cute,
        output_tensor_cute,
        local_buffer_ptr_cute,
        remote_buffer_ptr_cute,
        count_buffer_ptr_cute,
        recv_num_token_per_rank_cute,
        src_index_cute,
        src_expert_cute,
        src_offset_cute,
        src_rank_cute,
        src_token_cute,
        global_sync_semaphore_cute,
        current_stream,
    )

    # Distpatch
    # Input: 
    # - input_tensor: [num_tokens_per_rank, hidden_dim]
    # - topk_indices: [num_tokens_per_rank, num_topk]
    # Output:
    # - num_tokens_per_local_expert_recv: [num_local_experts, 1]
    # - dispatch_recv_token_tensor: [num_local_experts, num_tokens, 1, hidden_dim]

    intra_moe_kernel_compiled(
        input_tensor_cute,
        topk_indices_cute,
        num_tokens_per_local_expert_recv_cute,
        local_token_send_count_per_expert_cute,
        rank_token_count_cute,
        dispatch_recv_token_tensor_cute,
        combine_send_token_tensor_cute,
        output_tensor_cute,
        local_buffer_ptr_cute,
        remote_buffer_ptr_cute,
        count_buffer_ptr_cute,
        recv_num_token_per_rank_cute,
        src_index_cute,
        src_expert_cute,
        src_offset_cute,
        src_rank_cute,
        src_token_cute,
        global_sync_semaphore_cute,
        current_stream,
    )

    # torch.cuda.synchronize()

    # print("rank{}: Dispatch completed".format(rank))
    # print("rank{}: num_tokens_per_local_expert_recv: {}".format(rank, num_tokens_per_local_expert_recv))
    # print("rank{}: dispatch_recv_token_tensor shape: {}".format(rank, dispatch_recv_token_tensor.shape))

    return

    run_grouped_gemm(
        num_groups=2,
        problem_sizes_mnkl=((128, 128, 128, 1), (128, 128, 128, 1)),
        ab_dtype=cutlass.Float16,
        c_dtype=cutlass.Float16,
        acc_dtype=cutlass.Float32,
        a_major="k",
        b_major="k",
        c_major="n",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
        tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        tolerance=1e-01,
        warmup_iterations=0,
        iterations=1,
        skip_ref_check=False,
    )

    '''
    Combine
    '''

    '''
    Check results
    '''

    print(f"rank{rank}: PASS")


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
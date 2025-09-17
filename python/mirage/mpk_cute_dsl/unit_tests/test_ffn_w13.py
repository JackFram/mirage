import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import cutlass.utils as utils
import math

from cutlass.cute.runtime import from_dlpack
from cutlass.torch import dtype as torch_dtype

import mpk_cute_dsl.comm as comm

from mpk_cute_dsl.moe_utils import MoEParam
from mpk_cute_dsl.dist_utils import ProcessGroupInfo, parallel_launch
from mpk_cute_dsl.kernel.sm100_mpk_intra_moe import SM100MPKIntraMoEKernel
from mpk_cute_dsl.profiler.dsl_profiler import export_to_perfetto_trace
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKTask

def reset_tensors(dist_param: ProcessGroupInfo):
    torch.cuda.empty_cache()
        
    '''
    Initialize per rank input tensors
    '''

    num_ranks = dist_param.world_size

    num_tokens_per_rank, hidden_dim, inter_dim, num_topk, num_experts = 64, 5120, 2560, 8, 32

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

    # Meta Info tensors
    local_token_send_count_per_expert = torch.zeros((num_experts, 1), dtype=torch.int32, device='cuda')
    local_token_send_bar_expert = torch.zeros((num_experts, 1), dtype=torch.int32, device='cuda')
    rank_token_count = torch.zeros((1), dtype=torch.int32, device='cuda')
    recv_num_token_per_rank = torch.zeros((num_local_experts * num_ranks, 1), dtype=torch.int32, device='cuda')
    max_token_per_rank = min(num_local_experts * num_tokens, num_tokens * num_topk)
    src_index = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_expert = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_offset = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_rank = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    src_token = torch.zeros((max_token_per_rank), dtype=torch.int32, device='cuda')
    
    # MPK task queue buffer
    mpk_queue_len = 5120
    mpk_task_queue = torch.zeros((mpk_queue_len, 1), dtype=torch.uint32, device='cuda')
    mpk_task_consume_idx = torch.zeros((1), dtype=torch.int32, device='cuda')
    mpk_task_produce_idx = torch.zeros((1), dtype=torch.int32, device='cuda')
    mpk_task_barrier = torch.zeros((1024, 1), dtype=torch.int32, device='cuda')

    # Profiler tensor
    profiler_buffer_size = 148 * 9 * 128
    profiler_buffer = torch.zeros((profiler_buffer_size, 1), dtype=torch.uint64, device='cuda')
    profiler_ptr = torch.zeros((1), dtype=torch.uint32, device='cuda')

    '''
    dispatch
    '''
    # permute for group gemm
    permute_order = (1, 2, 0)

    dispatch_recv_token_tensor = torch.randn(
        (num_local_experts, num_tokens, hidden_dim),
        dtype=torch_dtype(moe_param.in_dtype),
    ).cuda().permute(permute_order) # also the input for the ffn fused w13 task
    
    num_tokens_per_local_expert_recv = torch.empty(
        (num_local_experts, 1),
        dtype=torch.int32,
    ).fill_(0).cuda()

    '''
    ffn_grouped_gemm
    '''
    w13_tensor = torch.randn(num_local_experts, hidden_dim, inter_dim * 2, dtype=torch_dtype(moe_param.out_dtype), device="cuda").permute(permute_order)
    w2_tensor = torch.randn(num_local_experts, inter_dim, hidden_dim, dtype=torch_dtype(moe_param.out_dtype), device="cuda").permute(permute_order)

    ffn_fused_w13_output_tensor = torch.empty(
        (num_local_experts, num_tokens, inter_dim),
        dtype=torch_dtype(moe_param.in_dtype),
    ).fill_(0).cuda().permute(permute_order)

    '''
    combine
    '''
    ffn_fused_w2_output_tensor = torch.randn(
        (num_local_experts, num_tokens, hidden_dim),
        dtype=torch_dtype(moe_param.out_dtype),
    ).fill_(0).cuda().permute(permute_order)
    
    combine_info_tensor = torch.empty(
        (num_local_experts, num_tokens, 1), # (activated-1b, source_rank-4b, source_expert-5b, source_index-8b)
        dtype=torch.uint32,
    ).fill_(0).cuda()

    # Initialize output tensors
    output_tensor = torch.empty(
        (num_tokens_per_rank, hidden_dim),
        dtype=torch_dtype(moe_param.out_dtype),
    ).fill_(0).cuda()

    # Convert tensors to cute tensors
    input_tensor_cute = from_dlpack(input_tensor, assumed_align=16)
    topk_indices_cute = from_dlpack(topk_indices, assumed_align=16)
    topk_weights_cute = from_dlpack(topk_weights, assumed_align=16)
    num_tokens_per_local_expert_recv_cute = from_dlpack(num_tokens_per_local_expert_recv, assumed_align=16)
    dispatch_recv_token_tensor_cute = from_dlpack(dispatch_recv_token_tensor, assumed_align=16)
    ffn_fused_w13_output_tensor_cute = from_dlpack(ffn_fused_w13_output_tensor, assumed_align=16)
    ffn_fused_w2_output_tensor_cute = from_dlpack(ffn_fused_w2_output_tensor, assumed_align=16)
    combine_info_tensor_cute = from_dlpack(combine_info_tensor, assumed_align=16)
    output_tensor_cute = from_dlpack(output_tensor, assumed_align=16)
    local_token_send_count_per_expert_cute = from_dlpack(local_token_send_count_per_expert, assumed_align=16)
    local_token_send_bar_expert_cute = from_dlpack(local_token_send_bar_expert, assumed_align=16)
    rank_token_count_cute = from_dlpack(rank_token_count, assumed_align=16)

    recv_num_token_per_rank_cute = from_dlpack(recv_num_token_per_rank, assumed_align=16)
    src_index_cute = from_dlpack(src_index, assumed_align=16)
    src_expert_cute = from_dlpack(src_expert, assumed_align=16)
    src_offset_cute = from_dlpack(src_offset, assumed_align=16)
    src_rank_cute = from_dlpack(src_rank, assumed_align=16)
    src_token_cute = from_dlpack(src_token, assumed_align=16)

    w13_tensor_cute = from_dlpack(w13_tensor, assumed_align=16)
    w2_tensor_cute = from_dlpack(w2_tensor, assumed_align=16)
    
    mpk_task_queue_cute = from_dlpack(mpk_task_queue, assumed_align=16)
    mpk_task_consume_idx_cute = from_dlpack(mpk_task_consume_idx, assumed_align=16)
    mpk_task_produce_idx_cute = from_dlpack(mpk_task_produce_idx, assumed_align=16)
    mpk_task_barrier_cute = from_dlpack(mpk_task_barrier, assumed_align=16)
    
    profiler_buffer_cute = from_dlpack(profiler_buffer, assumed_align=16)
    profiler_ptr_cute = from_dlpack(profiler_ptr, assumed_align=16)

    intra_moe_kernel = SM100MPKIntraMoEKernel(
        moe_param=moe_param,
        dist_param=dist_param,
        profiler_buffer_size=profiler_buffer_size,
        profiler_enabled=True,
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

    # setup kernel param
    kernel_param = MoEKernelParam(
            w13_tensor=w13_tensor_cute,
            w2_tensor=w2_tensor_cute,
            rank_input_tensor=input_tensor_cute,
            rank_input_topk_indices=topk_indices_cute,
            rank_input_topk_weights=topk_weights_cute,
            num_tokens_per_local_expert_recv=num_tokens_per_local_expert_recv_cute,
            local_token_send_count_per_expert=local_token_send_count_per_expert_cute,
            local_token_send_bar_expert=local_token_send_bar_expert_cute,
            rank_token_count=rank_token_count_cute,
            dispatch_recv_token_tensor=dispatch_recv_token_tensor_cute,
            ffn_fused_w13_output_tensor=ffn_fused_w13_output_tensor_cute,
            ffn_fused_w2_output_tensor=ffn_fused_w2_output_tensor_cute,
            combine_info_tensor=combine_info_tensor_cute,
            output_tensor=output_tensor_cute,
            local_buffer_ptr=local_buffer_ptr_cute,
            remote_buffer_ptr=remote_buffer_ptr_cute,
            count_buffer_ptr=count_buffer_ptr_cute,
            recv_num_token_per_rank=recv_num_token_per_rank_cute,
            src_index=src_index_cute,
            src_expert=src_expert_cute,
            src_offset=src_offset_cute,
            src_rank=src_rank_cute,
            src_token=src_token_cute,
            mpk_task_queue=mpk_task_queue_cute,
            mpk_task_consume_idx=mpk_task_consume_idx_cute,
            mpk_task_produce_idx=mpk_task_produce_idx_cute,
            mpk_task_barrier=mpk_task_barrier_cute,
        )
    
    """
    Reference implementation for ffn-w13 task here:
    """
    dispatch_recv_token_tensor_ref = dispatch_recv_token_tensor.clone()
    w13_tensor_ref = w13_tensor.clone()
    w13_output_ref = torch.einsum("mkl,nkl->mnl", dispatch_recv_token_tensor_ref, w13_tensor_ref)
    w1_output_ref = w13_output_ref[:, ::2, :]
    w3_output_ref = w13_output_ref[:, 1::2, :]
    ffn_fused_w13_output_tensor_ref = F.silu(w1_output_ref.to(torch.float32)) * w3_output_ref.to(torch.float32)
    ffn_fused_w13_output_tensor_ref = ffn_fused_w13_output_tensor_ref.to(ffn_fused_w13_output_tensor.dtype)
    """
    End of reference implementation
    """
    # MPK task initialization
    task_idx = 0
    for expert_idx in range(num_local_experts):
        for token_tile_idx in range(num_tokens // moe_param.mma_tiler_mn[1]):
            for ffn_task_id in range(inter_dim*2 // moe_param.mma_tiler_mn[0]): # two ffn fused w13 tasks
                mpk_task_queue[task_idx][0] = (MPKTask.kFusedFFNW13.value << 28) | (expert_idx << 16) | (token_tile_idx << 8) | ffn_task_id
                task_idx += 1
    # add termination task as we are testing ffn-w13 only.
    for i in range(148):
        mpk_task_queue[task_idx][0] = (MPKTask.kTerminate.value << 28)
        task_idx += 1
    
    mpk_task_produce_idx[0] = task_idx
    
    check_tensors = {
        "ffn_fused_w13_output_tensor": {
            "ref": ffn_fused_w13_output_tensor_ref,
            "cur": ffn_fused_w13_output_tensor,
            "ref_fn": lambda x: x,
            "cur_fn": lambda x: x,
            "atol": 1e-5,
            "rtol": 1e-5,
        }
    }

    return intra_moe_kernel, kernel_param, profiler_buffer_cute, profiler_ptr_cute, profiler_buffer, current_stream, check_tensors

def compile_kernel(dist_param: ProcessGroupInfo):

    intra_moe_kernel, kernel_param, profiler_buffer_cute, profiler_ptr_cute, profiler_buffer, current_stream, _ = reset_tensors(dist_param)
    torch.cuda.synchronize()
    
    intra_moe_kernel_compiled = cute.compile(
        intra_moe_kernel,
        kernel_param,
        profiler_buffer_cute,
        profiler_ptr_cute,
        current_stream,
    )
    
    return intra_moe_kernel_compiled

def test_dispatch(dist_param: ProcessGroupInfo, warm_up_iters=0, actual_iters=1):
    
    """
    Correctness alignment:
    """


    rank = dist_param.rank
    intra_moe_kernel_compiled = compile_kernel(dist_param)

    intra_moe_kernel, kernel_param, profiler_buffer_cute, profiler_ptr_cute, profiler_buffer, current_stream, check_tensors = reset_tensors(dist_param)
    torch.cuda.synchronize()

    intra_moe_kernel_compiled(
        kernel_param,
        profiler_buffer_cute,
        profiler_ptr_cute,
        current_stream,
    )
    
    torch.cuda.synchronize()
            
    for tensor_name in check_tensors:
        print(f"rank-{rank}, checking {tensor_name} ...")
        ref_fn = check_tensors[tensor_name]["ref_fn"]
        cur_fn = check_tensors[tensor_name]["cur_fn"]
        ref = ref_fn(check_tensors[tensor_name]["ref"]).cpu()
        cur = cur_fn(check_tensors[tensor_name]["cur"]).cpu()
        torch.testing.assert_close(
            cur,
            ref,
            atol=check_tensors[tensor_name]["atol"],
            rtol=check_tensors[tensor_name]["rtol"],
        )
        print(f"rank-{rank}, {tensor_name} check passed.")
        
    print(f"rank-{rank}, all check passed.")
    

if __name__ == "__main__":
    num_processes = 2
    parallel_launch(
        num_processes,
        test_dispatch,
    )
import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.const_param import ConstParam

from typing import List, Type, Union
from inspect import isclass

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class CombineRecvTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: cute.core.struct
        ):
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Combine-Recv"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "CombineRecvTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return CombineRecvTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Combine-Recv", event_type="begin")
        self.combine_recv()
        self.profiler.profile_event(event_name="Combine-Recv", event_type="end")
    
    @cute.jit
    def combine_recv(self):
        # Combine the received tokens from different ranks
        thread_idx, _, _ = cute.arch.thread_idx()
        token_id = (self.task_desc) & cutlass.Uint32(0x0000FFFF)
        
        local_rank = self.const_param.local_rank
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        ffn_w2_task_num = self.const_param.ffn_w2_task_num
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        num_worker_warps = self.const_param.num_worker_warps
        thr_tile_shape = self.const_param.thr_tile_shape
        num_topk = self.const_param.num_topk
        num_local_experts = self.const_param.num_local_experts
        moe_out_dtype = self.const_param.moe_out_dtype
        
        count_buffer_ptr = self.kernel_param.count_buffer_ptr
        local_buffer_ptr = self.kernel_param.local_buffer_ptr
        new_rank_input_topk_weights = self.kernel_param.rank_input_topk_weights
        output_tensor = self.kernel_param.output_tensor
        rank_input_topk_indices = self.kernel_param.rank_input_topk_indices

        if (thread_idx == 0):
            local_count_tensor = self.get_all_gather_count_buffer_ptr(
                count_buffer_ptr,
                local_rank,
                token_id,
            )

            # TODO(Zhihao): sometimes might hang here, figure out why
            count = inline_ptx.ld_flag_relaxed_sys_u32(local_count_tensor)
            while(cutlass.dynamic_expr(count != num_topk * ffn_w2_task_num)):
                count = inline_ptx.ld_flag_relaxed_sys_u32(local_count_tensor)
        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
        # reduction over received token 
        thr_tiled_output_tensor = cute.zipped_divide(output_tensor, thr_tile_shape)
        thr_dst_vec = thr_tiled_output_tensor[(None, (token_id, thread_idx))]
        
        # first item
        expert = rank_input_topk_indices[token_id, 0]
        src_rank = expert // num_local_experts
        src_local_expert_idx = expert % num_local_experts

        # Get the token pointer from the remote buffer
        token_tensor = self.get_combine_token_ptr_buffer(
            local_buffer_ptr,
            src_rank,
            src_local_expert_idx,
            token_id,
        )

        tiled_token_tensor = cute.zipped_divide(token_tensor, thr_tile_shape)
        thr_src_vec = tiled_token_tensor[(None, (0, thread_idx))]
        weight = new_rank_input_topk_weights[token_id, 0]
        acc_vec = thr_src_vec.load() * weight

        # remaining items
        for idx in cutlass.range_constexpr(1, num_topk, 1):
            expert = rank_input_topk_indices[token_id, idx]
            src_rank = expert // num_local_experts
            src_local_expert_idx = expert % num_local_experts

            # Get the token pointer from the remote buffer
            token_tensor = self.get_combine_token_ptr_buffer(
                local_buffer_ptr,
                src_rank,
                src_local_expert_idx,
                token_id,
            )

            tiled_token_tensor = cute.zipped_divide(token_tensor, thr_tile_shape)
            thr_src_vec = tiled_token_tensor[(None, (0, thread_idx))]
            weight = new_rank_input_topk_weights[token_id, idx]
            acc_vec += thr_src_vec.load() * weight

        thr_dst_vec.store(acc_vec.to(moe_out_dtype))

    @cute.jit
    def make_global_tensor_from_buffer_ptr(
        self,
        dtype: Type[cutlass.Numeric],
        offset: cutlass.Int64,
        layout: cutlass.cute.typing.Layout,
        ptr_i64: cutlass.Int64,
    ):
        """
        Create a global tensor from a buffer pointer.
        Args:
            dtype (Type[cutlass.Numeric]): The data type of the tensor.
            offset (cutlass.Int64): The offset in bytes of the tensor in the buffer.
            layout (cutlass.cute.typing.Layout): The layout of the tensor.
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
        Returns:
            cute.Tensor: The global tensor.
        """
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64+offset, cute.AddressSpace.gmem, assumed_align=16
        )
        tensor = cute.make_tensor(tensor_gmem_ptr, layout)
        return tensor
    
    def get_combine_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """
        moe_out_dtype = self.const_param.moe_out_dtype
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        token_buffer_offset_in_bytes = self.const_param.token_buffer_offset_in_bytes
        combine_token_stride = self.const_param.combine_token_stride
        hidden_dim = self.const_param.hidden_dim

        ptr_offset = token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * num_tokens_per_rank + recv_token_idx) * combine_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=moe_out_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
        
    def get_all_gather_count_buffer_ptr(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        index: cutlass.Int64 = 0,
    ):
        """
        Get the all gather count buffer pointer from the buffer pointer.
        Args:
            buffer_ptr_tensor (cute.Tensor): Tensor of buffer pointers.
            rank (cutlass.Int32): The rank of the pointer.
            index (cutlass.Int64): The index of the count buffer to access.
        Returns:
            cute.Tensor: The all gather count buffer pointer.
        """
        offset = index * 4
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=offset,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
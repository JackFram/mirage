import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from typing import List, Type, Union
from inspect import isclass

from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.const_param import ConstParam

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class DispatchSendTask:
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
        self.task_name = "Dispatch-Send"

        self.send_index_buffer = self.smem_storage.send_index_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DispatchSendTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return DispatchSendTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the dispatch send task
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="begin")
        self.dispatch_send()
        inline_ptx.fence_release_sys()
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="end")

    @cute.jit
    def dispatch_send(self):
        
        thread_idx, _, _ = cute.arch.thread_idx()

        token_idx = self.task_desc & cutlass.Uint32(0x0000007F)
        
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        thr_tile_shape = self.const_param.thr_tile_shape
        num_topk = self.const_param.num_topk
        num_worker_warps = self.const_param.num_worker_warps
        num_local_experts = self.const_param.num_local_experts

        remote_buffer_ptr = self.kernel_param.remote_buffer_ptr
        rank_input_tensor = self.kernel_param.rank_input_tensor
        rank_input_topk_indices = self.kernel_param.rank_input_topk_indices
        local_token_send_count_per_expert = self.kernel_param.local_token_send_count_per_expert
        local_token_send_bar_expert = self.kernel_param.local_token_send_bar_expert
        
        thr_tiled_rank_input_tensor = cute.zipped_divide(rank_input_tensor, thr_tile_shape)
        thr_src_vec = thr_tiled_rank_input_tensor[(None, (token_idx, thread_idx))]

        for topk_idx in cutlass.range_constexpr(0, num_topk, 1):

            # Get the local expert index
            expert_idx = rank_input_topk_indices[token_idx, topk_idx]
            
            # Get the synchronized index for sending tokens
            if (thread_idx == 0):
                recv_index = inline_ptx.atomic_add_flag_relaxed_gpu_global_u32(local_token_send_count_per_expert[expert_idx, None], 1)
                self.send_index_buffer[0] = recv_index
            cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
            remote_index = self.send_index_buffer[0]

            remote_rank = expert_idx // num_local_experts
            remote_expert_idx = expert_idx % num_local_experts

            remote_tensor = self.get_dispatch_token_ptr_buffer(
                remote_buffer_ptr,
                remote_rank,
                remote_expert_idx,
                remote_index,
            )
            
            meta_tensor = self.get_dispatch_meta_ptr_buffer(
                remote_buffer_ptr,
                remote_rank,
                remote_expert_idx,
                remote_index,
            )

            if (thread_idx == 0):
                # Store the meta data
                inline_ptx.st_flag_release_sys_global_u32(meta_tensor, token_idx)  # token index

            thr_tiled_rank_recv_tensor = cute.zipped_divide(remote_tensor, thr_tile_shape)
            thr_dst_vec = thr_tiled_rank_recv_tensor[(None, (0, thread_idx))]
            thr_dst_vec.store(thr_src_vec.load())

            cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
            
            # update the completion signal to a global sync buffer
            if (thread_idx == 0):
                # use release.sys to flush the data to gmem before updating the flag
                recv_index = inline_ptx.red_add_relaxed_sys_global_u32(local_token_send_bar_expert[expert_idx, None], 1)
            
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
    
    def get_dispatch_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
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

        hidden_dim = self.const_param.hidden_dim
        moe_in_dtype = self.const_param.moe_in_dtype
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        token_buffer_offset_in_bytes = self.const_param.token_buffer_offset_in_bytes
        dispatch_token_stride = self.const_param.dispatch_token_stride

        ptr_offset = token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * num_tokens_per_rank + recv_token_idx) * dispatch_token_stride

        # cute.printf(">?? rank: {}, expert_idx: {}, recv_token_idx: {}, offset: {}", rank, expert_idx, recv_token_idx, offset)
        # cute.printf(">?? token_buffer_offset_in_bytes: {}", self.token_buffer_offset_in_bytes)

        return self.make_global_tensor_from_buffer_ptr(
                dtype=moe_in_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
        
    def get_dispatch_meta_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the meta pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The meta pointer buffer.
        """

        hidden_dim_in_bytes = self.const_param.hidden_dim_in_bytes
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        token_buffer_offset_in_bytes = self.const_param.token_buffer_offset_in_bytes
        dispatch_token_stride = self.const_param.dispatch_token_stride

        ptr_offset = token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * num_tokens_per_rank + recv_token_idx) * dispatch_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=ptr_offset + hidden_dim_in_bytes,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
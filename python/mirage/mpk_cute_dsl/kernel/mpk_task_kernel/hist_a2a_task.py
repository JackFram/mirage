import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam

from typing import List, Type, Union
from inspect import isclass

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class HistAll2AllTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: SharedStorage
        ):
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Hist+All2All"
        
        self.expert_send_count = self.smem_storage.expert_send_count.get_tensor(
            cute.make_layout((32, 1), stride=(1, 1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "HistAll2AllTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return HistAll2AllTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the hist all-to-all task

        self.profiler.profile_event(event_name="Hist+All2All", event_type="begin")
        self.update_hist()
        self.profiler.profile_event(event_name="Hist+All2All", event_type="end")
        
    @cute.jit
    def update_hist(self):
        thread_idx, _, _ = cute.arch.thread_idx()
        # Update the histogram
        
        # const param
        num_worker_warps = self.const_param.num_worker_warps
        num_topk = self.const_param.num_topk
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        num_local_ranks = self.const_param.num_local_ranks
        num_local_experts = self.const_param.num_local_experts
        
        # kernel param
        rank_input_topk_indices = self.kernel_param.rank_input_topk_indices
        local_token_send_bar_expert = self.kernel_param.local_token_send_bar_expert
        worker_sync_bar_id = self.const_param.worker_sync_bar_id

        # comm param:
        remote_buffer_ptr = self.kernel_param.remote_buffer_ptr
        
        # init smem buffer
        for expert_idx in range(thread_idx, 32, num_worker_warps * 32):
            self.expert_send_count[expert_idx, 0] = 0
        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=self.const_param.num_worker_warps * 32)
        
        # get hist for each expert
        for element_idx in range(thread_idx, num_topk * num_tokens_per_rank, num_worker_warps * 32):
            expert_idx = rank_input_topk_indices[element_idx // num_topk, element_idx % num_topk]
            inline_ptx.red_add_shared_u32(self.expert_send_count[expert_idx, None], 1)
        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=self.const_param.num_worker_warps * 32)

        # notify remote rank for the completion of token transfer
        # TODO(Zhihao): currently we bind tidx to expert idx for synchronization, might result in warp divergence
        for expert_idx in range(thread_idx, num_local_ranks*num_local_experts, num_worker_warps * 32):
            expected_token_count = self.expert_send_count[expert_idx, 0]
            token_count = 0
            while(cutlass.dynamic_expr(token_count != expected_token_count)):
                token_count = inline_ptx.ld_flag_relaxed_gpu_u32(local_token_send_bar_expert[expert_idx, None])
            remote_rank = expert_idx // num_local_experts
            remote_expert_idx = expert_idx % num_local_experts
            sync_tensor = self.get_count_buffer_ptr(remote_buffer_ptr, remote_rank, remote_expert_idx)
            inline_ptx.st_flag_relaxed_sys_u32(sync_tensor, expected_token_count + 1)  # use the token count as the flag to indicate the dispatch send is done

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
    
    def get_count_buffer_ptr(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64 = 0,
    ):
        """
        Get the count buffer pointer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The count buffer pointer.
        """
        count_buffer_offset_in_bytes = self.const_param.count_buffer_offset_in_bytes

        offset = count_buffer_offset_in_bytes + expert_idx * 4
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=offset,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
import cutlass.cute as cute
import cutlass
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam

from typing import List, Type, Union
from inspect import isclass

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class DispatchRecvTask:
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
        self.task_name = "Dispatch-Recv"

        self.worker_sync_buffer = smem_storage.mpk_worker_sync_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DispatchRecvTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return DispatchRecvTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the dispatch send task
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        
        self.profiler.profile_event(event_name="Dispatch-Recv", event_type="begin")
        self.disptach_recv()
        self.profiler.profile_event(event_name="Dispatch-Recv", event_type="end")

    @cute.jit
    def disptach_recv(self):

        thread_idx, _, _ = cute.arch.thread_idx()
        # Decode the task description

        group_idx = self.task_desc & cutlass.Uint32(0x0000007F)

        local_buffer_ptr = self.kernel_param.local_buffer_ptr
        num_local_experts = self.const_param.num_local_experts
        num_worker_warps = self.const_param.num_worker_warps
        ffn_task_num = self.const_param.ffn_task_num

        mpk_task_produce_idx = self.kernel_param.mpk_task_produce_idx
        mpk_task_queue = self.kernel_param.mpk_task_queue
        mpk_queue_len = self.const_param.mpk_queue_len

        local_rank = group_idx // num_local_experts
        local_expert_idx = group_idx % num_local_experts

        if (thread_idx == 0): 
            count_tensor = self.get_count_buffer_ptr(local_buffer_ptr, local_rank, local_expert_idx)
            token_count = 0
            while(cutlass.dynamic_expr(token_count == 0)):
                token_count = inline_ptx.ld_flag_sys_acquire_u32(count_tensor)
            token_count -= 1
            self.worker_sync_buffer[0] = token_count

        cute.arch.barrier(barrier_id=0, number_of_threads=num_worker_warps * 32)
        token_count = self.worker_sync_buffer[0]
        
        if token_count > 0:
            # add fused ffn task to the queue
            for ffn_task_id in range(thread_idx, ffn_task_num, 32 * num_worker_warps):
                ffn_task_desc = cutlass.Uint32((4 << cutlass.Uint32(28)) | (group_idx << cutlass.Uint32(8)) | cutlass.Uint32(ffn_task_id))
                task_write_idx = inline_ptx.atomic_add(
                    mpk_task_produce_idx,
                    cutlass.Int32(1),
                ) % cutlass.Int32(mpk_queue_len)
                inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], ffn_task_desc)

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
import cutlass.cute as cute
import cutlass
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.const_param import ConstParam
from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKTask

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
            smem_storage: cute.core.struct
        ):
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Dispatch-Recv"

        self.worker_sync_buffer = smem_storage.mpk_worker_sync_buffer.get_tensor(
            cute.make_layout((16), stride=(1))
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

        group_idx = self.task_desc & cutlass.Uint32(0x00007FFF)

        num_local_experts = self.const_param.num_local_experts
        num_local_ranks = self.const_param.num_local_ranks
        num_worker_warps = self.const_param.num_worker_warps
        token_tile_size = self.const_param.token_tile_size
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        mpk_queue_len = self.const_param.mpk_queue_len
        tile_count_sync_id = self.const_param.tile_count_sync_offset
        
        local_buffer_ptr = self.kernel_param.local_buffer_ptr
        num_tokens_per_local_expert_recv = self.kernel_param.num_tokens_per_local_expert_recv
        rank_token_count = self.kernel_param.rank_token_count
        mpk_task_produce_idx = self.kernel_param.mpk_task_produce_idx
        mpk_task_queue = self.kernel_param.mpk_task_queue
        mpk_task_barrier = self.kernel_param.mpk_task_barrier 
        src_expert = self.kernel_param.src_expert
        src_offset = self.kernel_param.src_offset
        src_rank = self.kernel_param.src_rank
        src_token = self.kernel_param.src_token
        src_index = self.kernel_param.src_index

        local_rank = group_idx // num_local_experts
        local_expert_idx = group_idx % num_local_experts

        if (thread_idx == 0): 
            count_tensor = self.get_count_buffer_ptr(local_buffer_ptr, local_rank, local_expert_idx)
            token_count = 0
            while(cutlass.dynamic_expr(token_count == 0)):
                token_count = inline_ptx.ld_flag_sys_acquire_u32(count_tensor)
            token_count -= 1
            packed_count = (token_count << 16) | (0x00000001)
            accumulated_packed_count = inline_ptx.atomic_add(
                num_tokens_per_local_expert_recv[local_expert_idx, None],
                packed_count,
            ) 
            local_expert_offset = accumulated_packed_count >> 16
            arrived_group_count = (accumulated_packed_count & 0x0000FFFF) + 1
            
            # update the tile count in the barrier for synchronization with the complete of ffn later
            if arrived_group_count == num_local_ranks:
                tile_count = (local_expert_offset + token_count + token_tile_size - 1) // token_tile_size
                packed_tile_count = (tile_count << 24) | (1 << 16)
                inline_ptx.atomic_add(mpk_task_barrier[tile_count_sync_id, None], packed_tile_count)
            
            self.worker_sync_buffer[0] = token_count
            self.worker_sync_buffer[1] = local_expert_offset
            self.worker_sync_buffer[2] = inline_ptx.atomic_add(
                rank_token_count,
                token_count,
            )
            self.worker_sync_buffer[3] = arrived_group_count

        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)

        token_count = self.worker_sync_buffer[0]
        expert_start = self.worker_sync_buffer[1]
        token_start = self.worker_sync_buffer[2]
        arrived_group_count = self.worker_sync_buffer[3]

        for group_token_idx in range(thread_idx, token_count, num_worker_warps * 32):
            meta_tensor = self.get_dispatch_meta_ptr_buffer(
                local_buffer_ptr,
                local_rank,
                local_expert_idx,
                group_token_idx,
            )

            token_idx = token_start + group_token_idx  # absolute index of the token in the rank
            src_expert[token_idx] = local_expert_idx # relative expert index in the local rank
            src_offset[token_idx] = expert_start + group_token_idx # relative token index in the local expert
            src_rank[token_idx] = local_rank # relative rank index in the local world
            src_token[token_idx] = group_token_idx # relative token index in the local group ([local_rank, local_expert_idx])
            src_index[token_idx] = meta_tensor[0] # original token index in the input sequence

            last_tile_token_count = 0
            if arrived_group_count == num_local_ranks and group_token_idx == token_count - 1:
                last_tile_token_count = (expert_start + group_token_idx + 1) % token_tile_size # token count for the last ffn tile

            # add token gather task to the queue (one task per token)
            token_gather_desc = cutlass.Uint32((MPKTask.kTokenGather.value << cutlass.Uint32(28)) | (token_idx << cutlass.Uint32(12)) | cutlass.Uint32(last_tile_token_count))
            task_write_idx = inline_ptx.atomic_add(
                mpk_task_produce_idx,
                cutlass.Int32(1),
            ) % cutlass.Int32(mpk_queue_len)
            inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], token_gather_desc)
            
        # TODO(Zhihao): special case where the last arrived rank has no token at all (token_count == 0)

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
import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam
from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKTask

from typing import List, Type, Union
from inspect import isclass

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class TokenGatherTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: SharedStorage
        ):
        # Task Descripter Format:
        # | 31 - 28 |  27 - 12  |         11 - 0         |
        # | task_id | token_idx |  last_tile_token_count |
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Token-Gather-Task"
        
        self.worker_sync_buffer = smem_storage.mpk_worker_sync_buffer.get_tensor(
            cute.make_layout((16), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "TokenGatherTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return TokenGatherTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Token-Gather-Task", event_type="begin")
        self.token_gather()
        self.profiler.profile_event(event_name="Token-Gather-Task", event_type="end")

    @cute.jit
    def token_gather(self):
        thread_idx, _, _ = cute.arch.thread_idx()
        token_idx = (self.task_desc >> 12) & cutlass.Uint32(0x0000FFFF)
        last_tile_token_count = (self.task_desc) & cutlass.Uint32(0x00000FFF)
        
        num_worker_warps = self.const_param.num_worker_warps
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        thr_tile_shape = self.const_param.thr_tile_shape
        token_tile_per_expert = self.const_param.token_tile_per_expert
        token_tile_size = self.const_param.token_tile_size
        gemm_tile_bar_offset = self.const_param.gemm_tile_bar_offset
        ffn_w13_task_num = self.const_param.ffn_w13_task_num
        hidden_dim = self.const_param.hidden_dim

        src_expert = self.kernel_param.src_expert
        src_offset = self.kernel_param.src_offset
        src_rank = self.kernel_param.src_rank
        src_token = self.kernel_param.src_token
        local_buffer_ptr = self.kernel_param.local_buffer_ptr
        dispatch_recv_token_tensor = self.kernel_param.dispatch_recv_token_tensor
        mpk_task_barrier = self.kernel_param.mpk_task_barrier
        mpk_task_produce_idx = self.kernel_param.mpk_task_produce_idx
        mpk_task_queue = self.kernel_param.mpk_task_queue
        mpk_queue_len = self.const_param.mpk_queue_len

        # token gather kernel here
        
        dst_expert_offset = src_offset[token_idx]
        dst_expert = src_expert[token_idx]

        local_buffer_tensor = self.get_dispatch_token_ptr_buffer(
                local_buffer_ptr,
                src_rank[token_idx],
                dst_expert,
                src_token[token_idx],
            )
        tiled_src_tensor = cute.zipped_divide(local_buffer_tensor, thr_tile_shape)
        thr_src_vec = tiled_src_tensor[(None, (0, thread_idx))]

        dst_tensor = dispatch_recv_token_tensor[(dst_expert, dst_expert_offset, None)]
        # TODO(revisit): find a better way to deal with tile_divide if the overhead is non-negligible
        dst_tensor = cute.make_tensor(dst_tensor.iterator, cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)))
        # fix here by casting the layout
        tiled_dst_tensor = cute.zipped_divide(dst_tensor, thr_tile_shape)
        thr_dst_vec = tiled_dst_tensor[(None, (0, thread_idx))]
        thr_dst_vec.store(thr_src_vec.load())
        
        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
        
        if thread_idx == 0:
            token_tile_idx = dst_expert_offset // token_tile_size
            tile_group_sync_id = gemm_tile_bar_offset + dst_expert * token_tile_per_expert + token_tile_idx
            arrived_token_count = inline_ptx.atomic_add(mpk_task_barrier[tile_group_sync_id, None], 1) + 1
            self.worker_sync_buffer[0] = 0
            self.worker_sync_buffer[1] = token_tile_idx
            # normal tokens
            if last_tile_token_count == 0:
                if arrived_token_count == token_tile_size:
                    self.worker_sync_buffer[0] = 1
            # last token that launch the fused ffn task
            else:
                # cute.printf("expert-{}, relative offset-{}, tile_group_sync_id-{}, arrived_token_count-{}, expected_token_count-{}", dst_expert, dst_expert_offset, tile_group_sync_id, arrived_token_count, last_tile_token_count)
                while(cutlass.dynamic_expr(arrived_token_count != last_tile_token_count)):
                    arrived_token_count = inline_ptx.ld_flag_relaxed_gpu_u32(mpk_task_barrier[tile_group_sync_id, None])
                self.worker_sync_buffer[0] = 1

        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
        token_tile_idx = self.worker_sync_buffer[1]
        
        if self.worker_sync_buffer[0] == 1:
            # add fused ffn task to the queue
            for ffn_task_id in range(thread_idx, ffn_w13_task_num, 32 * num_worker_warps):
                ffn_task_desc = cutlass.Uint32((MPKTask.kFusedFFNW13.value << cutlass.Uint32(28)) | (dst_expert << cutlass.Uint32(16)) | (token_tile_idx << cutlass.Uint32(8)) | cutlass.Uint32(ffn_task_id))
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
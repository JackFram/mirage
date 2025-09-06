from typing import Optional, Union
import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from cutlass.cute.nvgpu import cpasync, tcgen05
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.const_param import ConstParam
from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKTask

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class FusedFFNW13Task:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: cute.core.struct
        ):
        # Task Descripter Format:
        # | 31 - 28 |   23 - 16  |  15 - 8  |    7 - 0    |
        # | task_id | expert_idx | tile_idx | ffn_task_id |
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Fused-FFN-W13-Task"
        
        self.worker_sync_buffer = smem_storage.mpk_worker_sync_buffer.get_tensor(
            cute.make_layout((16), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "FusedFFNW13Task":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return FusedFFNW13Task(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(
        self,
        w13_tma_atom_a: cute.CopyAtom,
        w13_tma_atom_b: cute.CopyAtom,
        w13_tma_atom_c: Optional[cute.CopyAtom],
    ):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Fused-FFN-W13", event_type="begin")
        self.fused_ffn_w13(
            w13_tma_atom_a=w13_tma_atom_a,
            w13_tma_atom_b=w13_tma_atom_b,
            w13_tma_atom_c=w13_tma_atom_c,
        )
        self.profiler.profile_event(event_name="Fused-FFN-W13", event_type="end")
    
    @cute.jit
    def fused_ffn_w13(
        self,
        w13_tma_atom_a: cute.CopyAtom,
        w13_tma_atom_b: cute.CopyAtom,
        w13_tma_atom_c: Optional[cute.CopyAtom],
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        expert_idx = (self.task_desc >> 16) & cutlass.Uint32(0x000000FF)
        tile_idx = (self.task_desc >> 8) & cutlass.Uint32(0x000000FF)
        ffn_w13_task_id = (self.task_desc) & cutlass.Uint32(0x000000FF)

        ffn_w2_bar_offset = self.const_param.ffn_w2_bar_offset
        token_tile_per_expert = self.const_param.token_tile_per_expert
        ffn_w13_task_num = self.const_param.ffn_w13_task_num
        ffn_w2_task_num = self.const_param.ffn_w2_task_num
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        num_worker_warps = self.const_param.num_worker_warps
        tile_count_sync_id = self.const_param.tile_count_sync_offset
        num_local_experts = self.const_param.num_local_experts
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        num_workers = self.const_param.num_workers
        tma_warp_id = self.const_param.tma_warp_id

        mpk_task_barrier = self.kernel_param.mpk_task_barrier
        mpk_task_produce_idx = self.kernel_param.mpk_task_produce_idx
        mpk_task_queue = self.kernel_param.mpk_task_queue
        mpk_queue_len = self.const_param.mpk_queue_len

        # fused kernel 
        
        # TODO(Zhihao): W1W3 pipelined accumulation + SwiGLU + element prod epilogue + SwapAB
        
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        # Prefetch tma desc
        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(w13_tma_atom_a)
            cpasync.prefetch_descriptor(w13_tma_atom_b)
            cpasync.prefetch_descriptor(w13_tma_atom_c)
        
        # no cluster launch and 2SM UMMA
        mma_tile_coord_v = 0
        is_leader_cta = True
        cta_rank_in_cluster = 0
        block_in_cluster_coord_vmnk = (0, 0, 0, 0)
        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #

        tmem_dealloc_mbar_ptr = self.smem_storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = self.smem_storage.tmem_holding_buf

        # TODO(Zhihao): continue here

        # end of the kernel
        
        if thread_idx == 0:
            tile_group_sync_id = ffn_w2_bar_offset + expert_idx * token_tile_per_expert + tile_idx
            arrived_tile_count = inline_ptx.atomic_add(mpk_task_barrier[tile_group_sync_id, None], 1) + 1
            self.worker_sync_buffer[0] = arrived_tile_count
            
        cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
        arrived_tile_count = self.worker_sync_buffer[0]
        
        if arrived_tile_count == ffn_w13_task_num:
            # add fused_ffn_w2 task to the task queue
            for ffn_w2_task_id in range(thread_idx, ffn_w2_task_num, 32 * num_worker_warps):
                ffn_task_desc = cutlass.Uint32((MPKTask.kFusedFFNW2Send.value << cutlass.Uint32(28)) | (expert_idx << cutlass.Uint32(16)) | (tile_idx << cutlass.Uint32(8)) | cutlass.Uint32(ffn_w2_task_id))
                task_write_idx = inline_ptx.atomic_add(
                    mpk_task_produce_idx,
                    cutlass.Int32(1),
                ) % cutlass.Int32(mpk_queue_len)
                inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], ffn_task_desc)
            
            cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
            
            if thread_idx == 0:
                packed_value = inline_ptx.atomic_add(mpk_task_barrier[tile_count_sync_id, None], 1)
                total_tile_count = packed_value >> 24
                arrived_expert_count = (packed_value >> 16) & 0x000000FF
                arrived_tile_count = (packed_value & 0x0000FFFF) + 1
                
                self.worker_sync_buffer[1] = 0
                if arrived_expert_count == num_local_experts and arrived_tile_count == total_tile_count:
                    self.worker_sync_buffer[1] = 1
                    
            cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
            # last fused_ffn_w2 task added, now add combine_recv task
            if self.worker_sync_buffer[1] == 1:
                for token_idx in range(thread_idx, num_tokens_per_rank, 32 * num_worker_warps):
                    combine_recv_task_desc = cutlass.Uint32((MPKTask.kCombineRecv.value << cutlass.Uint32(28)) | cutlass.Uint32(token_idx))
                    task_write_idx = inline_ptx.atomic_add(
                        mpk_task_produce_idx,
                        cutlass.Int32(1),
                    ) % cutlass.Int32(mpk_queue_len)
                    inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], combine_recv_task_desc)
                cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)
                for sm_idx in range(thread_idx, num_workers, 32 * num_worker_warps):
                    terminate_task_desc = cutlass.Uint32((MPKTask.kTerminate.value << cutlass.Uint32(28)))
                    task_write_idx = inline_ptx.atomic_add(
                        mpk_task_produce_idx,
                        cutlass.Int32(1),
                    ) % cutlass.Int32(mpk_queue_len)
                    inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], terminate_task_desc)
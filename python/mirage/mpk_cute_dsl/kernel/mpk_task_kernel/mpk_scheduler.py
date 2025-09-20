from typing import Optional, Union
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir import ir

from mpk_cute_dsl.kernel.mpk_task_kernel import *

import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam

from mpk_cute_dsl.const_param import ConstParam
from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKTask

# Task Descripter Format:
# | 31 - 28 | 27 - 0 |
# | task_id |  meta  |

class MPKScheduler:
    def __init__(
            self, 
            scheduler_warp_idx: cutlass.Constexpr[int], 
            smem_storage: cute.core.struct,
            const_param: ConstParam,
            kernel_param: MoEKernelParam,
            profiler: DslProfiler,
        ):
        self.task_desc = cutlass.Uint32(0)
        self.scheduler_warp_idx = scheduler_warp_idx
        self.smem_storage = smem_storage
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.profiler = profiler

        self.task_sync_buffer = smem_storage.mpk_task_sync_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

    def __extract_mlir_values__(self):
        # TODO(revisit): do we need to add MLIR for all members here?
        profiler_values = extract_mlir_values(self.profiler)
        values = [profiler_values[0][0], profiler_values[1][0]]
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MPKScheduler":
        assert len(values) == 2
        # new_profiler = new_from_mlir_values(
        #     self.profiler, [values[0], values[1]]
        # )
        return MPKScheduler(
            self.scheduler_warp_idx,
            self.smem_storage,
            self.const_param,
            self.kernel_param,
            self.profiler
        )

    @cute.jit
    def fetch_next_task(self):
        # gmem -> register task load with atomic add
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()

        mpk_queue_len = self.const_param.mpk_queue_len
        task_consume_idx = self.kernel_param.mpk_task_consume_idx
        task_queue = self.kernel_param.mpk_task_queue

        # specialized scheduler warp 
        if warp_idx == self.scheduler_warp_idx:
            self.profiler.profile_event(event_name="Fetch-Task", event_type="begin")
            if thread_idx == self.scheduler_warp_idx * 32:

                # self.profiler.profile_event(event_name="Fetch-Task", event_type="begin")
                task_load_idx = inline_ptx.atomic_add(
                    task_consume_idx,
                    cutlass.Int32(1),
                ) % cutlass.Int32(mpk_queue_len)
                # peek
                task_desc = inline_ptx.ld_flag_relaxed_gpu_u32(task_queue[task_load_idx, None])
                task_code = task_desc >> 28
                # prefetch next task
                while(cutlass.dynamic_expr(task_code == 0)): 
                    # Wait task update if task_code == 0 (fetch)
                    # TODO(Zhihao): try ld.relax and also measure the overhead (might slow down works on other warps)
                    task_desc = inline_ptx.ld_flag_relaxed_gpu_u32(task_queue[task_load_idx, None])
                    task_code = task_desc >> 28
                    inline_ptx.nanosleep(100)
                # register -> smem task store from scheduler warp
                self.task_sync_buffer[0] = task_desc
            cute.arch.sync_warp()
            self.profiler.profile_event(event_name="Fetch-Task", event_type="end") 

    @cute.jit
    def add_task(self, task_desc: cutlass.Uint32):
        # register -> gmem task store with atomic add
        mpk_queue_len = self.const_param.mpk_queue_len
        thread_idx, _, _ = cute.arch.thread_idx()
        task_produce_idx = self.kernel_param.mpk_task_produce_idx
        task_queue = self.kernel_param.mpk_task_queue
        if thread_idx == self.scheduler_warp_idx * 32:
            self.profiler.profile_event(event_name="Add-Task", event_type="begin")
            task_write_idx = inline_ptx.atomic_add(
                task_produce_idx,
                cutlass.Int32(1),
            ) % cutlass.Int32(mpk_queue_len)
            inline_ptx.st_flag_volatile(task_queue[task_write_idx, None], task_desc)
            self.profiler.profile_event(event_name="Add-Task", event_type="end")

    @cute.jit
    def sync_task(self):
        # smem -> register task load from worker warp
        thread_idx, _, _ = cute.arch.thread_idx()
        self.profiler.profile_event(event_name="Sync-Task", event_type="begin")
        cute.arch.sync_threads()
        self.task_desc = self.task_sync_buffer[0].to(cutlass.Uint32)
        self.profiler.profile_event(event_name="Sync-Task", event_type="end")
        
    @cute.jit
    def execute_task(
        self,
        tiled_mma: cute.TiledMma,
        w13_d_tiled_mma: cute.TiledMma,
        w13_tma_atom_a: cute.CopyAtom,
        w13_tma_atom_b: cute.CopyAtom,
        w13_tma_atom_c: cute.CopyAtom,
        w2_tma_atom_a: cute.CopyAtom,
        w2_tma_atom_b: cute.CopyAtom,
        w13_mA_mkl: cute.Tensor,
        w13_mB_nkl: cute.Tensor,
        w2_mA_mkl: cute.Tensor,
        w2_mB_nkl: cute.Tensor,
        w13_mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w2_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        w13_c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        w13_d_tile: cute.Tile,
        w2_d_tile: cute.Tile,
    ):
        # task decode
        # device kernel execution with switch
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()

        task_code = (self.task_desc >> 28)
        is_final_task = (task_code == MPKTask.kTerminate.value)

        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        num_worker_warps = self.const_param.num_worker_warps
        
        mpk_task_barrier = self.kernel_param.mpk_task_barrier

        # executing task with worker warps
        if not is_final_task and warp_idx < self.scheduler_warp_idx:

            if task_code == MPKTask.kFetch.value:
                task_runner = FetchTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kHistAll2All.value:
                task_runner = HistAll2AllTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kDispatchSend.value:
                task_runner = DispatchSendTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kDispatchRecv.value:
                task_runner = DispatchRecvTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kFusedFFNW13.value:
                task_runner = FusedFFNW13Task(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute(
                    tiled_mma=tiled_mma,
                    w13_d_tiled_mma=w13_d_tiled_mma,
                    w13_tma_atom_a=w13_tma_atom_a,
                    w13_tma_atom_b=w13_tma_atom_b,
                    w13_tma_atom_c=w13_tma_atom_c,
                    w13_mA_mkl=w13_mA_mkl,
                    w13_mB_nkl=w13_mB_nkl,
                    w13_mC_mnl=w13_mC_mnl,
                    cluster_layout_vmnk=cluster_layout_vmnk,
                    a_smem_layout_staged=a_smem_layout_staged,
                    b_smem_layout_staged=b_smem_layout_staged,
                    c_smem_layout_staged=w13_c_smem_layout_staged,
                    epi_tile=epi_tile,
                    w13_d_tile=w13_d_tile,
                )
            elif task_code == MPKTask.kFusedFFNW2Send.value:
                task_runner = FusedFFNW2SendTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute(
                    tiled_mma=tiled_mma,
                    w2_tma_atom_a=w2_tma_atom_a,
                    w2_tma_atom_b=w2_tma_atom_b,
                    w2_mA_mkl=w2_mA_mkl,
                    w2_mB_nkl=w2_mB_nkl,
                    cluster_layout_vmnk=cluster_layout_vmnk,
                    a_smem_layout_staged=a_smem_layout_staged,
                    b_smem_layout_staged=b_smem_layout_staged,
                    c_smem_layout_staged=w2_c_smem_layout_staged,
                    epi_tile=epi_tile,
                    w2_d_tile=w2_d_tile,
                )
            elif task_code == MPKTask.kCombineRecv.value:
                task_runner = CombineRecvTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kTokenGather.value:
                task_runner = TokenGatherTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
                
            cute.arch.barrier(barrier_id=worker_sync_bar_id, number_of_threads=num_worker_warps * 32)

        # if thread_idx == 0:
        #     cute.printf("Block {} finished task {}", block_idx, task_code)

        return is_final_task
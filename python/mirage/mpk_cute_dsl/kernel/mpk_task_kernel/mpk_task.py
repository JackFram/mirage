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
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam

from enum import Enum

class MPKTask(Enum):
    kFetch: cutlass.Uint32 = 0
    kHistAll2All: cutlass.Uint32 = 1
    kDispatchSend: cutlass.Uint32 = 2
    kDispatchRecv: cutlass.Uint32 = 3
    kCombineSend: cutlass.Uint32 = 4
    kCombineRecv: cutlass.Uint32 = 5
    kTerminate: cutlass.Uint32 = 6

# TODO(Zhihao): add task desc structure
# | 31 - 28 | 27 - 16 | 15 - 0 |
# |  task   |  param  | token id|

class MPKScheduler:
    def __init__(
            self, 
            scheduler_warp_idx: cutlass.Constexpr[int], 
            task_queue: cute.Tensor, 
            task_consume_idx: cute.Tensor, 
            task_produce_idx: cute.Tensor, 
            task_barrier: cute.Tensor,
            smem_storage: SharedStorage,
            const_param: ConstParam,
            kernel_param: MoEKernelParam,
            profiler: DslProfiler,
        ):
        self.task_desc = cutlass.Uint32(0)
        self.scheduler_warp_idx = scheduler_warp_idx
        self.task_queue = task_queue
        self.task_consume_idx = task_consume_idx
        self.task_produce_idx = task_produce_idx
        self.task_barrier = task_barrier
        self.smem_storage = smem_storage
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.profiler = profiler

        self.task_sync_buffer = smem_storage.mpk_task_sync_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_queue.__extract_mlir_values__()
        values.extend(self.task_consume_idx.__extract_mlir_values__())
        values.extend(self.task_produce_idx.__extract_mlir_values__())
        values.extend(self.task_barrier.__extract_mlir_values__())
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MPKScheduler":
        assert len(values) == 4
        new_task_queue = new_from_mlir_values(
            self.task_queue, [values[0]]
        )
        new_task_consume_idx = new_from_mlir_values(
            self.task_consume_idx, [values[1]]
        )
        new_task_produce_idx = new_from_mlir_values(
            self.task_produce_idx, [values[2]]
        )
        new_task_barrier = new_from_mlir_values(
            self.task_barrier, [values[3]]
        )
        return MPKScheduler(
            self.scheduler_warp_idx,
            new_task_queue,
            new_task_consume_idx,
            new_task_produce_idx,
            new_task_barrier,
            self.smem_storage,
            self.const_param,
            self.kernel_param,
            self.profiler,
        )

    @cute.jit
    def fetch_next_task(self):
        # gmem -> register task load with atomic add
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        thread_idx, _, _ = cute.arch.thread_idx()
        # specialized scheduler warp 
        if warp_idx == self.scheduler_warp_idx:
            self.profiler.profile_event(event_name="Fetch-Task", event_type="begin")
            if thread_idx == self.scheduler_warp_idx * 32:

                # self.profiler.profile_event(event_name="Fetch-Task", event_type="begin")
                task_load_idx = inline_ptx.atomic_add(
                    self.task_consume_idx,
                    cutlass.Int32(1),
                ) % cutlass.Int32(1024)
                task_desc = inline_ptx.ld_flag_volatile(self.task_queue[task_load_idx, None])
                # task_code = 0
                # while(cutlass.dynamic_expr(task_code == 0)): 
                #     # Wait task update if task_code == 0 (fetch)
                #     # TODO(Zhihao): try ld.relax and also measure the overhead (might slow down works on other warps)
                #     task_desc = inline_ptx.ld_flag_volatile(self.task_queue[task_load_idx, None])
                #     task_code = task_desc >> 28
                #     break
                # register -> smem task store from scheduler warp
                self.task_sync_buffer[0] = task_desc
            self.profiler.profile_event(event_name="Fetch-Task", event_type="end")

    @cute.jit
    def add_task(self, task_desc: cutlass.Uint32):
        # register -> gmem task store with atomic add
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx == self.scheduler_warp_idx * 32:
            self.profiler.profile_event(event_name="Add-Task", event_type="begin")
            task_write_idx = inline_ptx.atomic_add(
                self.task_produce_idx,
                cutlass.Int32(1),
            ) % cutlass.Int32(1024)
            inline_ptx.st_flag_volatile(self.task_queue[task_write_idx, None], task_desc)
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
    def execute_task(self):
        # task decode
        # device kernel execution with switch
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()

        task_code = (self.task_desc >> 28)

        # executing task with worker warps
        if warp_idx < self.scheduler_warp_idx:

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
            elif task_code == MPKTask.kCombineSend.value:
                task_runner = CombineSendTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()
            elif task_code == MPKTask.kCombineRecv.value:
                task_runner = CombineRecvTask(self.task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)
                task_runner.execute()

        is_final_task = (task_code == MPKTask.kTerminate.value)

        return is_final_task
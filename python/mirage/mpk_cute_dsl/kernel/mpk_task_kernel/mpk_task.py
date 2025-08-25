import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir import ir

from mpk_cute_dsl.kernel.mpk_task_kernel.hist_a2a_task import HistAll2AllTask
from mpk_cute_dsl.kernel.mpk_task_kernel.dispatch_send_task import DispatchSendTask
from mpk_cute_dsl.kernel.mpk_task_kernel.dispatch_recv_task import DispatchRecvTask
from mpk_cute_dsl.kernel.mpk_task_kernel.combine_send_task import CombineSendTask
from mpk_cute_dsl.kernel.mpk_task_kernel.combine_recv_task import CombineRecvTask

import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

from enum import Enum

class MPKTask(Enum):
    kFetch: cutlass.Int32 = 0
    kHistAll2All: cutlass.Int32 = 1
    kDispatchSend: cutlass.Int32 = 2
    kDispatchRecv: cutlass.Int32 = 3
    kCombineSend: cutlass.Int32 = 4
    kCombineRecv: cutlass.Int32 = 5
    kTerminate: cutlass.Int32 = 6

class MPKScheduler:
    def __init__(
            self, 
            scheduler_warp_idx:cute.Int32, 
            task_queue:cute.Tensor, 
            task_consume_idx:cute.Tensor, 
            task_produce_idx:cute.Tensor, 
            task_barrier:cute.Tensor,
            task_sync_buffer:cute.Tensor,
            profiler: DslProfiler,
        ):
        self.task_desc = cutlass.Int32(0)
        self.scheduler_warp_idx = scheduler_warp_idx
        self.task_queue = task_queue
        self.task_consume_idx = task_consume_idx
        self.task_produce_idx = task_produce_idx
        self.task_barrier = task_barrier
        self.task_sync_buffer = task_sync_buffer
        self.profiler = profiler
        
    def __extract_mlir_values__(self):
        values = self.task_queue.__extract_mlir_values__()
        values.extend(self.task_consume_idx.__extract_mlir_values__())
        values.extend(self.task_produce_idx.__extract_mlir_values__())
        values.extend(self.task_barrier.__extract_mlir_values__())
        values.extend(self.task_sync_buffer.__extract_mlir_values__())
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MPKScheduler":
        assert len(values) == 5
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
        new_task_sync_buffer = new_from_mlir_values(
            self.task_sync_buffer, [values[4]]
        )
        return MPKScheduler(
            self.scheduler_warp_idx,
            new_task_queue,
            new_task_consume_idx,
            new_task_produce_idx,
            new_task_barrier,
            new_task_sync_buffer,
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
                self.task_sync_buffer[0] = task_desc
            self.profiler.profile_event(event_name="Fetch-Task", event_type="end")

    @cute.jit
    def add_task(self, task_desc: cutlass.Int32):
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
        # register -> smem task store from scheduler warp
        # smem -> register task load from worker warp
        thread_idx, _, _ = cute.arch.thread_idx()
        self.profiler.profile_event(event_name="Sync-Task", event_type="begin")
        cute.arch.sync_threads()
        self.task_desc = self.task_sync_buffer[0]
        self.profiler.profile_event(event_name="Sync-Task", event_type="end")
        
    @cute.jit
    def execute_task(self):
        # task decode
        # device kernel execution with switch
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        
        task_code = self.task_desc >> 28
        # For debug
        if thread_idx == 0:
            cute.printf("block-{} recv task: {}", block_idx, task_code)
        return
        # End debug
        
        # if task_code == MPKTask.kTerminate.value:
        #     return
        
        # executing task with worker warps
        if warp_idx < self.scheduler_warp_idx:
            task_runner = decode_task(task_code, self.task_desc, self.profiler)
            task_runner.execute()

@cute.jit
def decode_task(task_code: cutlass.Int32, task_desc: cutlass.Int32, profiler: DslProfiler=None):
    # task_id:4b|depend_flag:1b|sync_buffer_idx:3x4b|meta:15b
    if task_code == MPKTask.kFetch.value:
        raise ValueError("Task code 0 is reserved for fetch operations and should not be executed directly.")
    elif task_code == MPKTask.kHistAll2All.value:
        return HistAll2AllTask(task_desc, profiler)
    elif task_code == MPKTask.kDispatchSend.value:
        return DispatchSendTask(task_desc, profiler)
    elif task_code == MPKTask.kDispatchRecv.value:
        return DispatchRecvTask(task_desc, profiler)
    elif task_code == MPKTask.kCombineSend.value:
        return CombineSendTask(task_desc, profiler)
    elif task_code == MPKTask.kCombineRecv.value:
        return CombineRecvTask(task_desc, profiler)
    else:
        raise ValueError(f"Unknown task code: {task_code}")

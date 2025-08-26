import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class DispatchSendTask:
    def __init__(self, task_desc: cutlass.Uint32, profiler: DslProfiler):
        self.task_desc = task_desc
        self.profiler = profiler
        self.task_name = "Dispatch-Send"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DispatchSendTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return DispatchSendTask(new_task_desc, self.profiler)

    @cute.jit
    def execute(self):
        # Execute the dispatch send task
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()

        token_idx = self.task_desc & cutlass.Uint32(0x0000FFFF)
        
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="begin")
        ### task implementation
        inline_ptx.nanosleep(50*1000)
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="end")
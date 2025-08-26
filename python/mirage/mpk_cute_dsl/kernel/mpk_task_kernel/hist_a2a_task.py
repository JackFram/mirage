import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class HistAll2AllTask:
    def __init__(self, task_desc: cutlass.Uint32, profiler: DslProfiler):
        self.task_desc = task_desc
        self.profiler = profiler
        self.task_name = "Hist+All2All"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "HistAll2AllTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return HistAll2AllTask(new_task_desc, self.profiler)

    @cute.jit
    def execute(self):
        # Execute the hist all-to-all task
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()

        self.profiler.profile_event(event_name="Hist+All2All", event_type="begin")
        ### task implementation
        inline_ptx.nanosleep(50*1000)
        self.profiler.profile_event(event_name="Hist+All2All", event_type="end")
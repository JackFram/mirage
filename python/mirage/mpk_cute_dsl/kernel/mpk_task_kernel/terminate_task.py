import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class TerminateTask:
    def __init__(self, task_desc: cutlass.Uint32, profiler: DslProfiler):
        self.task_desc = task_desc
        self.profiler = profiler
        self.task_name = "Terminate"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "TerminateTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return TerminateTask(new_task_desc, self.profiler)

    @cute.jit
    def execute(self):
        self.profiler.profile_event(event_name="Terminate-Task", event_type="instant")
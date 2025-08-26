import cutlass.cute as cute
import cutlass
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class DispatchRecvTask:
    def __init__(self, task_desc: cutlass.Uint32, profiler: DslProfiler):
        self.task_desc = task_desc
        self.profiler = profiler
        self.task_name = "Dispatch-Recv"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DispatchRecvTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return DispatchRecvTask(new_task_desc, self.profiler)

    @cute.jit
    def execute(self):
        # Execute the dispatch receive task
        pass
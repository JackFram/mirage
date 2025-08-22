import cutlass.cute as cute
import cutlass
from .base_task import BaseTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

class DispatchRecvTask(BaseTask):
    def __init__(self, task_desc: cutlass.Int32, profiler: DslProfiler = None):
        super().__init__(task_desc, profiler, "Dispatch-Recv")

    @cute.jit
    def execute(self):
        # Execute the dispatch receive task
        pass
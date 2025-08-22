import cutlass.cute as cute
import cutlass
from .base_task import BaseTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

class CombineSendTask(BaseTask):
    def __init__(self, task_desc: cutlass.Int32, profiler: DslProfiler = None):
        super().__init__(task_desc, profiler, "Combine-Send")

    @cute.jit
    def execute(self):
        # Execute the combine send task
        pass
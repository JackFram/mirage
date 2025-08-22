import cutlass
import cutlass.cute as cute
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

class BaseTask:
    def __init__(self, task_desc: cutlass.Int32, profiler: DslProfiler = None, task_name: str = None):
        self.task_desc = task_desc
        self.profiler = profiler
        self.task_name = task_name

    @cute.jit
    def execute(self):
        # Execute the task
        raise NotImplementedError("Subclasses should implement this method.")
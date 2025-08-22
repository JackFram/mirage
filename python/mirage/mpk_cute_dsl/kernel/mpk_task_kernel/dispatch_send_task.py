import cutlass.cute as cute
import cutlass
from .base_task import BaseTask
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

class DispatchSendTask(BaseTask):
    def __init__(self, task_desc: cutlass.Int32, profiler: DslProfiler = None):
        super().__init__(task_desc, profiler, "Dispatch-Send")

    @cute.jit
    def execute(self):
        # Execute the dispatch send task
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        if thread_idx == 0:
            print("block_idx[{}]-executing task: {}".format(block_idx, self.task_desc))

        self.profiler.profile_event(event_name="Dispatch-Send", event_type="begin")
        ### task implementation
        inline_ptx.nanosleep(50*1000)
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="end")
import cutlass.cute as cute
import cutlass
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class FusedFFNTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: SharedStorage
        ):
        # Task Descripter Format:
        # | 31 - 28 |     27      |  26 - 15   |   14 - 8  |    7 - 0    |
        # | task_id | depend_flag | barrier_id | group_idx | ffn_task_id |
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Fused-FFN-Task"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "FusedFFNTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return FusedFFNTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Fused-FFN", event_type="begin")
        self.fused_ffn()
        self.profiler.profile_event(event_name="Fused-FFN", event_type="end")
    
    @cute.jit
    def fused_ffn(self):
        pass
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

class CombineRecvTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: SharedStorage
        ):
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Combine-Recv"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "CombineRecvTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return CombineRecvTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Combine-Recv", event_type="begin")
        self.combine_recv()
        self.profiler.profile_event(event_name="Combine-Recv", event_type="end")
    
    @cute.jit
    def combine_recv(self):
        # Combine the received tokens from different ranks
        thread_idx, _, _ = cute.arch.thread_idx()
        token_id = (self.task_desc) & cutlass.Uint32(0x0000FFFF)
        pass
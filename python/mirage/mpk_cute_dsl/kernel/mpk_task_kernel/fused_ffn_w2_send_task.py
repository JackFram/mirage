import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam

from mpk_cute_dsl.const_param import ConstParam

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class FusedFFNW2SendTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: cute.core.struct
        ):
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Fused-FFN-W2-Send-Task"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "FusedFFNW2SendTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return FusedFFNW2SendTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Fused-FFN-W2-Send", event_type="begin")
        self.fused_ffn_w2()
        self.profiler.profile_event(event_name="Fused-FFN-W2-Send", event_type="end")

    @cute.jit
    def fused_ffn_w2(self):
        thread_idx, _, _ = cute.arch.thread_idx()
        expert_idx = (self.task_desc >> 16) & cutlass.Uint32(0x000000FF)
        tile_idx = (self.task_desc >> 8) & cutlass.Uint32(0x000000FF)
        ffn_w2_task_id = (self.task_desc) & cutlass.Uint32(0x000000FF)

        ffn_w2_bar_offset = self.const_param.ffn_w2_bar_offset
        token_tile_per_expert = self.const_param.token_tile_per_expert
        ffn_w2_task_num = self.const_param.ffn_w2_task_num
        worker_sync_bar_id = self.const_param.worker_sync_bar_id
        num_worker_warps = self.const_param.num_worker_warps
        
        mpk_task_barrier = self.kernel_param.mpk_task_barrier
        mpk_task_produce_idx = self.kernel_param.mpk_task_produce_idx
        mpk_task_queue = self.kernel_param.mpk_task_queue
        mpk_queue_len = self.const_param.mpk_queue_len

        # fused kernel 
        
        # TODO(Zhihao): W2 GeMM
        
        # end of the kernel
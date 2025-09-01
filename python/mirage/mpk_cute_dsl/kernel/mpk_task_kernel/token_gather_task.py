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

class TokenGatherTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: SharedStorage
        ):
        # Task Descripter Format:
        # | 31 - 28 |  27 - 12  |         11 - 0         |
        # | task_id | token_idx |  last_tile_token_count |
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Token-Gather-Task"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "TokenGatherTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return TokenGatherTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Token-Gather-Task", event_type="begin")
        self.token_gather()
        self.profiler.profile_event(event_name="Token-Gather-Task", event_type="end")

    @cute.jit
    def token_gather(self):
        thread_idx, _, _ = cute.arch.thread_idx()
        token_idx = (self.task_desc >> 12) & cutlass.Uint32(0x0000FFFF)
        last_tile_token_count = (self.task_desc) & cutlass.Uint32(0x00000FFF)
        
        # if thread_idx == 0:
        #     cute.printf("token-{}, last_tile_token_count-{}", token_idx, last_tile_token_count)

        # token gather kernel here
        
        # end of the kernel
        
        # # add fused ffn task to the queue
        # for ffn_task_id in range(thread_idx, ffn_task_num, 32 * num_worker_warps):
        #     ffn_task_desc = cutlass.Uint32((MPKTask.kFusedFFNW13.value << cutlass.Uint32(28)) | (group_idx << cutlass.Uint32(8)) | cutlass.Uint32(ffn_task_id))
        #     task_write_idx = inline_ptx.atomic_add(
        #         mpk_task_produce_idx,
        #         cutlass.Int32(1),
        #     ) % cutlass.Int32(mpk_queue_len)
        #     inline_ptx.st_flag_relaxed_gpu_u32(mpk_task_queue[task_write_idx, None], ffn_task_desc)

import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.kernel.mpk_task_kernel.smem_storage import SharedStorage
from mpk_cute_dsl.const_param import ConstParam

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class DispatchSendTask:
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
        self.task_name = "Dispatch-Send"

        self.send_index_buffer = self.smem_storage.send_index_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "DispatchSendTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return DispatchSendTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(self):
        # Execute the dispatch send task
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="begin")
        self.dispatch_send()
        self.profiler.profile_event(event_name="Dispatch-Send", event_type="end")

    @cute.jit
    def dispatch_send(self):
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        block_dim, _, _ = cute.arch.block_dim()
        num_sm, _, _ = cute.arch.grid_dim()

        token_idx = self.task_desc & cutlass.Uint32(0x0000FFFF)

        remote_buffer_ptr = self.kernel_param.remote_buffer_ptr
        rank_input_tensor = self.kernel_param.rank_input_tensor
        rank_input_topk_indices = self.kernel_param.rank_input_topk_indices
        local_token_send_count_per_expert = self.kernel_param.local_token_send_count_per_expert

        thr_tiled_rank_input_tensor = cute.zipped_divide(rank_input_tensor, self.const_param.thr_tile_shape)
        thr_src_vec = thr_tiled_rank_input_tensor[(None, (token_idx, thread_idx))]

        for topk_idx in cutlass.range_constexpr(0, self.const_param.num_topk, 1):

            # Get the local expert index
            expert_idx = rank_input_topk_indices[token_idx, topk_idx]
            
            # Get the synchronized index for sending tokens
            if (thread_idx == 0):
                recv_index = inline_ptx.atomic_add(local_token_send_count_per_expert[expert_idx, None], 1)
                self.send_index_buffer[0] = recv_index
            cute.arch.barrier(barrier_id=0, number_of_threads=self.const_param.num_worker_warps * 32)
            remote_index = self.send_index_buffer[0]

            if thread_idx == 0:
                cute.printf("remote index-{}", remote_index)
            
            # TODO(continue)

        #     remote_rank = expert_idx // self.num_local_experts
        #     remote_expert_idx = expert_idx % self.num_local_experts

        #     remote_tensor = self.get_dispatch_token_ptr_buffer(
        #         remote_buffer_ptr,
        #         remote_rank,
        #         remote_expert_idx,
        #         remote_index,
        #     )

        #     meta_tensor = self.get_dispatch_meta_ptr_buffer(
        #         remote_buffer_ptr,
        #         remote_rank,
        #         remote_expert_idx,
        #         remote_index,
        #     )

        #     if (thread_idx == 0):
        #         # Store the meta data
        #         meta_tensor[0] = cutlass.Int32(token_idx)  # token index


        #     thr_tiled_rank_recv_tensor = cute.zipped_divide(remote_tensor, self.thr_tile_shape)
        #     thr_dst_vec = thr_tiled_rank_recv_tensor[(None, (0, thread_idx))]
                
        #     thr_dst_vec.store(thr_src_vec.load())

        #     cute.arch.sync_threads()

        # # grid_sync

        # self.grid_sync()

        # # send token count to remote buffer

        # for expert_idx in range(block_idx * block_dim + thread_idx, self.moe_param.num_experts, num_sm * block_dim):
        #     remote_rank = expert_idx // self.num_local_experts
        #     remote_expert_idx = expert_idx % self.num_local_experts
        #     sync_tensor = self.get_count_buffer_ptr(remote_buffer_ptr, remote_rank, remote_expert_idx)
        #     inline_ptx.st_flag_release(sync_tensor, local_token_send_count_per_expert[expert_idx, 0] + 1)  # use the token count as the flag to indicate the dispatch send is done
        #     # cute.printf(">??-[send-{}] remote_rank: {}, remote_expert_idx: {}, token_count: {}", self.dist_param.local_rank, remote_rank, remote_expert_idx, local_token_send_count_per_expert[expert_idx])
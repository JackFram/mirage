import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import (
    get_c_pointers,
    get_mlir_types,
    extract_mlir_values,
    new_from_mlir_values,
)
from cutlass._mlir import ir

class MoEKernelParam:
    def __init__(
            self,
            # weight tensors
            w13_tensor: cute.Tensor,
            w2_tensor: cute.Tensor,
            # input/output tensors
            rank_input_tensor: cute.Tensor,
            rank_input_topk_indices: cute.Tensor,
            num_tokens_per_local_expert_recv: cute.Tensor,
            local_token_send_count_per_expert: cute.Tensor,
            local_token_send_bar_expert: cute.Tensor,
            rank_token_count: cute.Tensor,
            dispatch_recv_token_tensor: cute.Tensor,
            ffn_fused_w13_output_tensor: cute.Tensor,
            combine_send_token_tensor: cute.Tensor,
            output_tensor: cute.Tensor,
            # buffer ptr
            local_buffer_ptr: cute.Tensor,
            remote_buffer_ptr: cute.Tensor,
            count_buffer_ptr: cute.Tensor,
            # meta info tensors
            recv_num_token_per_rank: cute.Tensor,
            src_index: cute.Tensor, 
            src_expert: cute.Tensor,
            src_offset: cute.Tensor,
            src_rank: cute.Tensor,
            src_token: cute.Tensor,
            # mpk queue tensor
            mpk_task_queue: cute.Tensor,
            mpk_task_consume_idx: cute.Tensor,
            mpk_task_produce_idx: cute.Tensor,
            mpk_task_barrier: cute.Tensor,
    ):
        self.w13_tensor = w13_tensor
        self.w2_tensor = w2_tensor
        self.rank_input_tensor = rank_input_tensor
        self.rank_input_topk_indices = rank_input_topk_indices
        self.num_tokens_per_local_expert_recv = num_tokens_per_local_expert_recv
        self.local_token_send_count_per_expert = local_token_send_count_per_expert
        self.local_token_send_bar_expert = local_token_send_bar_expert
        self.rank_token_count = rank_token_count
        self.dispatch_recv_token_tensor = dispatch_recv_token_tensor
        self.ffn_fused_w13_output_tensor = ffn_fused_w13_output_tensor
        self.combine_send_token_tensor = combine_send_token_tensor
        self.output_tensor = output_tensor
        self.local_buffer_ptr = local_buffer_ptr
        self.remote_buffer_ptr = remote_buffer_ptr
        self.count_buffer_ptr = count_buffer_ptr
        self.recv_num_token_per_rank = recv_num_token_per_rank
        self.src_index = src_index
        self.src_expert = src_expert
        self.src_offset = src_offset
        self.src_rank = src_rank
        self.src_token = src_token
        self.mpk_task_queue = mpk_task_queue
        self.mpk_task_consume_idx = mpk_task_consume_idx
        self.mpk_task_produce_idx = mpk_task_produce_idx
        self.mpk_task_barrier = mpk_task_barrier

    def __c_pointers__(self):
        pointers = []
        pointers.extend(get_c_pointers(self.w13_tensor))
        pointers.extend(get_c_pointers(self.w2_tensor))
        pointers.extend(get_c_pointers(self.rank_input_tensor))
        pointers.extend(get_c_pointers(self.rank_input_topk_indices))
        pointers.extend(get_c_pointers(self.num_tokens_per_local_expert_recv))  
        pointers.extend(get_c_pointers(self.local_token_send_count_per_expert))
        pointers.extend(get_c_pointers(self.local_token_send_bar_expert))
        pointers.extend(get_c_pointers(self.rank_token_count))
        pointers.extend(get_c_pointers(self.dispatch_recv_token_tensor))
        pointers.extend(get_c_pointers(self.ffn_fused_w13_output_tensor))
        pointers.extend(get_c_pointers(self.combine_send_token_tensor))
        pointers.extend(get_c_pointers(self.output_tensor))
        pointers.extend(get_c_pointers(self.local_buffer_ptr))
        pointers.extend(get_c_pointers(self.remote_buffer_ptr))
        pointers.extend(get_c_pointers(self.count_buffer_ptr))
        pointers.extend(get_c_pointers(self.recv_num_token_per_rank))
        pointers.extend(get_c_pointers(self.src_index))
        pointers.extend(get_c_pointers(self.src_expert))
        pointers.extend(get_c_pointers(self.src_offset))
        pointers.extend(get_c_pointers(self.src_rank))
        pointers.extend(get_c_pointers(self.src_token))
        pointers.extend(get_c_pointers(self.mpk_task_queue))
        pointers.extend(get_c_pointers(self.mpk_task_consume_idx))
        pointers.extend(get_c_pointers(self.mpk_task_produce_idx))
        pointers.extend(get_c_pointers(self.mpk_task_barrier))
        return pointers

    def __extract_mlir_values__(self):
        values = []
        values.extend(extract_mlir_values(self.w13_tensor))
        values.extend(extract_mlir_values(self.w2_tensor))
        values.extend(extract_mlir_values(self.rank_input_tensor))
        values.extend(extract_mlir_values(self.rank_input_topk_indices))
        values.extend(extract_mlir_values(self.num_tokens_per_local_expert_recv))
        values.extend(extract_mlir_values(self.local_token_send_count_per_expert))
        values.extend(extract_mlir_values(self.local_token_send_bar_expert))
        values.extend(extract_mlir_values(self.rank_token_count))
        values.extend(extract_mlir_values(self.dispatch_recv_token_tensor))
        values.extend(extract_mlir_values(self.ffn_fused_w13_output_tensor))
        values.extend(extract_mlir_values(self.combine_send_token_tensor))
        values.extend(extract_mlir_values(self.output_tensor))
        values.extend(extract_mlir_values(self.local_buffer_ptr))
        values.extend(extract_mlir_values(self.remote_buffer_ptr))
        values.extend(extract_mlir_values(self.count_buffer_ptr))
        values.extend(extract_mlir_values(self.recv_num_token_per_rank))
        values.extend(extract_mlir_values(self.src_index))
        values.extend(extract_mlir_values(self.src_expert))
        values.extend(extract_mlir_values(self.src_offset))
        values.extend(extract_mlir_values(self.src_rank))
        values.extend(extract_mlir_values(self.src_token))
        values.extend(extract_mlir_values(self.mpk_task_queue))
        values.extend(extract_mlir_values(self.mpk_task_consume_idx))
        values.extend(extract_mlir_values(self.mpk_task_produce_idx))
        values.extend(extract_mlir_values(self.mpk_task_barrier))
        return values
    
    def __get_mlir_types__(self):
        values = []
        values.extend(get_mlir_types(self.w13_tensor))
        values.extend(get_mlir_types(self.w2_tensor))
        values.extend(get_mlir_types(self.rank_input_tensor))
        values.extend(get_mlir_types(self.rank_input_topk_indices))
        values.extend(get_mlir_types(self.num_tokens_per_local_expert_recv))
        values.extend(get_mlir_types(self.local_token_send_count_per_expert))
        values.extend(get_mlir_types(self.local_token_send_bar_expert))
        values.extend(get_mlir_types(self.rank_token_count))
        values.extend(get_mlir_types(self.dispatch_recv_token_tensor))
        values.extend(get_mlir_types(self.ffn_fused_w13_output_tensor))
        values.extend(get_mlir_types(self.combine_send_token_tensor))
        values.extend(get_mlir_types(self.output_tensor))
        values.extend(get_mlir_types(self.local_buffer_ptr))
        values.extend(get_mlir_types(self.remote_buffer_ptr))
        values.extend(get_mlir_types(self.count_buffer_ptr))
        values.extend(get_mlir_types(self.recv_num_token_per_rank))
        values.extend(get_mlir_types(self.src_index))
        values.extend(get_mlir_types(self.src_expert))
        values.extend(get_mlir_types(self.src_offset))
        values.extend(get_mlir_types(self.src_rank))
        values.extend(get_mlir_types(self.src_token))
        values.extend(get_mlir_types(self.mpk_task_queue))
        values.extend(get_mlir_types(self.mpk_task_consume_idx))
        values.extend(get_mlir_types(self.mpk_task_produce_idx))
        values.extend(get_mlir_types(self.mpk_task_barrier))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MoEKernelParam":
        assert len(values) == 25
        value_idx = 0
        new_w13_tensor = new_from_mlir_values(
            self.w13_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_w2_tensor = new_from_mlir_values(
            self.w2_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_rank_input_tensor = new_from_mlir_values(
            self.rank_input_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_rank_input_topk_indices = new_from_mlir_values(
            self.rank_input_topk_indices, [values[value_idx]]
        )
        value_idx += 1
        new_num_tokens_per_local_expert_recv = new_from_mlir_values(
            self.num_tokens_per_local_expert_recv, [values[value_idx]]
        )
        value_idx += 1
        new_local_token_send_count_per_expert = new_from_mlir_values(
            self.local_token_send_count_per_expert, [values[value_idx]]
        )
        value_idx += 1
        new_local_token_send_bar_expert = new_from_mlir_values(
            self.local_token_send_bar_expert, [values[value_idx]]
        )
        value_idx += 1
        new_rank_token_count = new_from_mlir_values(
            self.rank_token_count, [values[value_idx]]
        )
        value_idx += 1
        new_dispatch_recv_token_tensor = new_from_mlir_values(
            self.dispatch_recv_token_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_ffn_fused_w13_output_tensor = new_from_mlir_values(
            self.ffn_fused_w13_output_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_combine_send_token_tensor = new_from_mlir_values(
            self.combine_send_token_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_output_tensor = new_from_mlir_values(
            self.output_tensor, [values[value_idx]]
        )
        value_idx += 1
        new_local_buffer_ptr = new_from_mlir_values(
            self.local_buffer_ptr, [values[value_idx]]
        )
        value_idx += 1
        new_remote_buffer_ptr = new_from_mlir_values(
            self.remote_buffer_ptr, [values[value_idx]]
        )
        value_idx += 1
        new_count_buffer_ptr = new_from_mlir_values(
            self.count_buffer_ptr, [values[value_idx]]
        )
        value_idx += 1
        new_recv_num_token_per_rank = new_from_mlir_values(
            self.recv_num_token_per_rank, [values[value_idx]]
        )
        value_idx += 1
        new_src_index = new_from_mlir_values(
            self.src_index, [values[value_idx]]
        )
        value_idx += 1
        new_src_expert = new_from_mlir_values(
            self.src_expert, [values[value_idx]]
        )
        value_idx += 1
        new_src_offset = new_from_mlir_values(
            self.src_offset, [values[value_idx]]
        )
        value_idx += 1
        new_src_rank = new_from_mlir_values(
            self.src_rank, [values[value_idx]]
        )
        value_idx += 1
        new_src_token = new_from_mlir_values(
            self.src_token, [values[value_idx]]
        )
        value_idx += 1
        new_mpk_task_queue = new_from_mlir_values(
            self.mpk_task_queue, [values[value_idx]]
        )
        value_idx += 1
        new_mpk_task_consume_idx = new_from_mlir_values(
            self.mpk_task_consume_idx, [values[value_idx]]
        )
        value_idx += 1
        new_mpk_task_produce_idx = new_from_mlir_values(
            self.mpk_task_produce_idx, [values[value_idx]]
        )
        value_idx += 1
        new_mpk_task_barrier = new_from_mlir_values(
            self.mpk_task_barrier, [values[value_idx]]
        )
        return MoEKernelParam(
            new_w13_tensor,
            new_w2_tensor,
            new_rank_input_tensor,
            new_rank_input_topk_indices,
            new_num_tokens_per_local_expert_recv,
            new_local_token_send_count_per_expert,
            new_local_token_send_bar_expert,
            new_rank_token_count,
            new_dispatch_recv_token_tensor,
            new_ffn_fused_w13_output_tensor,
            new_combine_send_token_tensor,
            new_output_tensor,
            new_local_buffer_ptr,
            new_remote_buffer_ptr,
            new_count_buffer_ptr,
            new_recv_num_token_per_rank,
            new_src_index,
            new_src_expert,
            new_src_offset,
            new_src_rank,
            new_src_token,
            new_mpk_task_queue,
            new_mpk_task_consume_idx,
            new_mpk_task_produce_idx,
            new_mpk_task_barrier,
        )
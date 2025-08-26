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
            # input tensors
            rank_input_tensor: cute.Tensor,
            rank_input_topk_indices: cute.Tensor,
            # output tensor
            num_tokens_per_local_expert_recv: cute.Tensor,
            local_token_send_count_per_expert: cute.Tensor,
            rank_token_count: cute.Tensor,
            dispatch_recv_token_tensor: cute.Tensor,
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
    ):
        self.rank_input_tensor = rank_input_tensor
        self.rank_input_topk_indices = rank_input_topk_indices
        self.num_tokens_per_local_expert_recv = num_tokens_per_local_expert_recv
        self.local_token_send_count_per_expert = local_token_send_count_per_expert
        self.rank_token_count = rank_token_count
        self.dispatch_recv_token_tensor = dispatch_recv_token_tensor
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

    def __c_pointers__(self):
        pointers = []
        pointers.extend(get_c_pointers(self.rank_input_tensor))
        pointers.extend(get_c_pointers(self.rank_input_topk_indices))
        pointers.extend(get_c_pointers(self.num_tokens_per_local_expert_recv))  
        pointers.extend(get_c_pointers(self.local_token_send_count_per_expert))
        pointers.extend(get_c_pointers(self.rank_token_count))
        pointers.extend(get_c_pointers(self.dispatch_recv_token_tensor))
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
        return pointers

    def __extract_mlir_values__(self):
        values = []
        values.extend(extract_mlir_values(self.rank_input_tensor))
        values.extend(extract_mlir_values(self.rank_input_topk_indices))
        values.extend(extract_mlir_values(self.num_tokens_per_local_expert_recv))
        values.extend(extract_mlir_values(self.local_token_send_count_per_expert))
        values.extend(extract_mlir_values(self.rank_token_count))
        values.extend(extract_mlir_values(self.dispatch_recv_token_tensor))
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
        return values
    
    def __get_mlir_types__(self):
        values = []
        values.extend(get_mlir_types(self.rank_input_tensor))
        values.extend(get_mlir_types(self.rank_input_topk_indices))
        values.extend(get_mlir_types(self.num_tokens_per_local_expert_recv))
        values.extend(get_mlir_types(self.local_token_send_count_per_expert))
        values.extend(get_mlir_types(self.rank_token_count))
        values.extend(get_mlir_types(self.dispatch_recv_token_tensor))
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
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "MoEKernelParam":
        assert len(values) == 17
        new_rank_input_tensor = new_from_mlir_values(
            self.rank_input_tensor, [values[0]]
        )
        new_rank_input_topk_indices = new_from_mlir_values(
            self.rank_input_topk_indices, [values[1]]
        )
        new_num_tokens_per_local_expert_recv = new_from_mlir_values(
            self.num_tokens_per_local_expert_recv, [values[2]]
        )
        new_local_token_send_count_per_expert = new_from_mlir_values(
            self.local_token_send_count_per_expert, [values[3]]
        )
        new_rank_token_count = new_from_mlir_values(
            self.rank_token_count, [values[4]]
        )
        new_dispatch_recv_token_tensor = new_from_mlir_values(
            self.dispatch_recv_token_tensor, [values[5]]
        )
        new_combine_send_token_tensor = new_from_mlir_values(
            self.combine_send_token_tensor, [values[6]]
        )
        new_output_tensor = new_from_mlir_values(
            self.output_tensor, [values[7]]
        )
        new_local_buffer_ptr = new_from_mlir_values(
            self.local_buffer_ptr, [values[8]]
        )
        new_remote_buffer_ptr = new_from_mlir_values(
            self.remote_buffer_ptr, [values[9]]
        )
        new_count_buffer_ptr = new_from_mlir_values(
            self.count_buffer_ptr, [values[10]]
        )
        new_recv_num_token_per_rank = new_from_mlir_values(
            self.recv_num_token_per_rank, [values[11]]
        )
        new_src_index = new_from_mlir_values(
            self.src_index, [values[12]]
        )
        new_src_expert = new_from_mlir_values(
            self.src_expert, [values[13]]
        )
        new_src_offset = new_from_mlir_values(
            self.src_offset, [values[14]]
        )
        new_src_rank = new_from_mlir_values(
            self.src_rank, [values[15]]
        )
        new_src_token = new_from_mlir_values(
            self.src_token, [values[16]]
        )
        return MoEKernelParam(
            new_rank_input_tensor,
            new_rank_input_topk_indices,
            new_num_tokens_per_local_expert_recv,
            new_local_token_send_count_per_expert,
            new_rank_token_count,
            new_dispatch_recv_token_tensor,
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
        )
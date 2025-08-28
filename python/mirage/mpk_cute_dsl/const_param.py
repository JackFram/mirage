import cutlass
import cutlass.cute as cute
import torch

class ConstParam:
    def __init__(
            self, 
            hidden_dim: cutlass.Constexpr[int],
            hidden_dim_in_bytes: cutlass.Constexpr[int],
            moe_in_dtype: cutlass.Constexpr[torch.dtype],
            num_topk: cutlass.Constexpr[int],
            num_tokens_per_rank: cutlass.Constexpr[int],
            num_local_experts: cutlass.Constexpr[int],
            num_local_ranks: cutlass.Constexpr[int],
            local_rank: cutlass.Constexpr[int],
            token_buffer_offset_in_bytes: cutlass.Constexpr[int],
            count_buffer_offset_in_bytes: cutlass.Constexpr[int],
            dispatch_token_stride: cutlass.Constexpr[int],
            num_worker_warps: cutlass.Constexpr[int],
            thr_tile_shape: tuple[int, int]
        ):
        
        # kernel const parameters
        self.num_worker_warps = num_worker_warps
        self.thr_tile_shape = thr_tile_shape
        
        # moe comm buffer const parameters
        self.token_buffer_offset_in_bytes = token_buffer_offset_in_bytes
        self.count_buffer_offset_in_bytes = count_buffer_offset_in_bytes
        self.dispatch_token_stride = dispatch_token_stride

        # moe const parameters
        self.hidden_dim = hidden_dim
        self.hidden_dim_in_bytes = hidden_dim_in_bytes
        self.moe_in_dtype = moe_in_dtype
        self.num_topk = num_topk
        self.num_tokens_per_rank = num_tokens_per_rank
        self.num_local_experts = num_local_experts

        # dist const parameters
        self.num_local_ranks = num_local_ranks
        self.local_rank = local_rank
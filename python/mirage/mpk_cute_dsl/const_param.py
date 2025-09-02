import cutlass
import cutlass.cute as cute
import torch
import math

class ConstParam:
    def __init__(
            self, 
            hidden_dim: cutlass.Constexpr[int],
            hidden_dim_in_bytes: cutlass.Constexpr[int],
            inter_dim: cutlass.Constexpr[int],
            moe_in_dtype: cutlass.Constexpr[torch.dtype],
            num_topk: cutlass.Constexpr[int],
            num_tokens_per_rank: cutlass.Constexpr[int],
            num_local_experts: cutlass.Constexpr[int],
            num_local_ranks: cutlass.Constexpr[int],
            local_rank: cutlass.Constexpr[int],
            token_buffer_offset_in_bytes: cutlass.Constexpr[int],
            count_buffer_offset_in_bytes: cutlass.Constexpr[int],
            dispatch_token_stride: cutlass.Constexpr[int],
            mpk_queue_len: cutlass.Constexpr[int],
            num_worker_warps: cutlass.Constexpr[int],
            thr_tile_shape: tuple[int, int],
            mma_tiler_mn: tuple[int, int],
            swapAB: bool,
            num_workers: cutlass.Constexpr[int],
        ):
        
        # kernel const parameters
        self.num_workers = num_workers
        self.num_worker_warps = num_worker_warps
        self.thr_tile_shape = thr_tile_shape
        self.mma_tiler_mn = mma_tiler_mn
        self.swapAB = swapAB
        self.mpk_queue_len = mpk_queue_len
        self.token_tile_size = self.mma_tiler_mn[0] if not swapAB else self.mma_tiler_mn[1]
        self.k_tile_size = self.mma_tiler_mn[1] if not swapAB else self.mma_tiler_mn[0]
        self.worker_sync_bar_id = 1
        self.token_tile_per_expert = math.ceil((num_tokens_per_rank * num_local_ranks) / self.token_tile_size)
        self.ffn_w13_task_num = math.ceil(inter_dim / self.k_tile_size)
        self.ffn_w2_task_num = math.ceil(hidden_dim / self.k_tile_size)

        # moe comm buffer const parameters
        self.token_buffer_offset_in_bytes = token_buffer_offset_in_bytes
        self.count_buffer_offset_in_bytes = count_buffer_offset_in_bytes
        self.dispatch_token_stride = dispatch_token_stride

        # moe const parameters
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.hidden_dim_in_bytes = hidden_dim_in_bytes
        self.moe_in_dtype = moe_in_dtype
        self.num_topk = num_topk
        self.num_tokens_per_rank = num_tokens_per_rank
        self.num_local_experts = num_local_experts

        # dist const parameters
        self.num_local_ranks = num_local_ranks
        self.local_rank = local_rank

        # barrier offsets
        self.gemm_tile_bar_offset = 0 # self.token_tile_per_expert * num_local_experts
        self.ffn_w2_bar_offset = self.gemm_tile_bar_offset + self.token_tile_per_expert * num_local_experts # self.token_tile_per_expert * num_local_experts
        self.tile_count_sync_offset = self.ffn_w2_bar_offset + self.token_tile_per_expert * num_local_experts # 1
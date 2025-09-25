import cutlass
import cutlass.cute as cute
import torch
import math
from typing import Optional, Type, Union
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils as utils
from cutlass.torch import dtype as torch_dtype

class ConstParam:
    def __init__(
            self, 
            # MoE param
            hidden_dim: cutlass.Constexpr[int],
            hidden_dim_in_bytes: cutlass.Constexpr[int],
            inter_dim: cutlass.Constexpr[int],
            moe_in_dtype: cutlass.Constexpr[torch.dtype],
            moe_out_dtype: cutlass.Constexpr[torch.dtype],
            num_topk: cutlass.Constexpr[int],
            num_tokens_per_rank: cutlass.Constexpr[int],
            num_local_experts: cutlass.Constexpr[int],
            num_local_ranks: cutlass.Constexpr[int],
            local_rank: cutlass.Constexpr[int],
            token_buffer_offset_in_bytes: cutlass.Constexpr[int],
            count_buffer_offset_in_bytes: cutlass.Constexpr[int],
            dispatch_token_stride: cutlass.Constexpr[int],
            combine_token_stride: cutlass.Constexpr[int],
            # mpk param
            mpk_queue_len: cutlass.Constexpr[int],
            num_worker_warps: cutlass.Constexpr[int],
            num_workers: cutlass.Constexpr[int],
            # gemm param
            c_dtype: cutlass.Constexpr[cutlass.Numeric],
            c_layout: cutlass.Constexpr[utils.LayoutEnum],
            thr_tile_shape: tuple[int, int],
            mma_tiler: tuple[int, int, int],
            d_mma_tiler: tuple[int, int, int],
            cta_tile_shape_mnk: tuple[int, int, int],
            occupancy: cutlass.Constexpr[int],
            swapAB: bool,
            acc_dtype: cutlass.Constexpr[cutlass.Numeric],
            use_2cta_instrs: bool,
            cluster_shape_mn: tuple[int, int],
            num_tma_load_bytes: cutlass.Constexpr[int],
            num_tmem_alloc_cols: cutlass.Constexpr[int],
            # smem stage
            num_ab_stage: cutlass.Constexpr[int],
            num_c_stage: cutlass.Constexpr[int],
            num_acc_stage: cutlass.Constexpr[int],
        ):
        
        # kernel const parameters
        self.num_workers = num_workers
        self.num_worker_warps = num_worker_warps
        self.thr_tile_shape = thr_tile_shape
        self.swapAB = swapAB
        self.mpk_queue_len = mpk_queue_len
        self.token_tile_size = mma_tiler[0] if not swapAB else mma_tiler[1]
        self.w_tile_size = mma_tiler[1] if not swapAB else mma_tiler[0]
        self.k_tile_size = mma_tiler[2]
        self.worker_sync_bar_id = 1
        self.token_tile_per_expert = math.ceil((num_tokens_per_rank * num_local_ranks) / self.token_tile_size)
        self.ffn_w13_task_num = math.ceil(2 * inter_dim / self.w_tile_size)
        self.ffn_w2_task_num = math.ceil(hidden_dim / self.w_tile_size)
        self.ffn_w13_k_cnt = math.ceil(hidden_dim / self.k_tile_size)
        self.ffn_w2_k_cnt = math.ceil(inter_dim / self.k_tile_size)

        # gemm const parameters
        self.c_dtype = c_dtype
        self.c_layout = c_layout
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = mma_tiler
        self.d_mma_tiler = d_mma_tiler
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = occupancy

        self.epilog_warp_id = (
            0,
            1,
            2,
            3,
        )
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.scheduler_warp_id = 8

        self.num_tmem_alloc_cols = num_tmem_alloc_cols

        # smem stage:
        self.num_ab_stage = num_ab_stage
        self.num_c_stage = num_c_stage
        self.num_acc_stage = num_acc_stage

        print(self.num_ab_stage, self.num_c_stage, self.num_acc_stage)

        # Set barrier id for cta sync, epilog sync, tmem ptr sync and tensormap update sync
        self.cta_sync_bar_id = 2
        self.epilog_sync_bar_id = 3
        self.tmem_ptr_sync_bar_id = 4
        self.tensormap_ab_init_bar_id = 5
        self.num_tma_load_bytes = num_tma_load_bytes

        # moe comm buffer const parameters
        self.token_buffer_offset_in_bytes = token_buffer_offset_in_bytes
        self.count_buffer_offset_in_bytes = count_buffer_offset_in_bytes
        self.dispatch_token_stride = dispatch_token_stride
        self.combine_token_stride = combine_token_stride

        # moe const parameters
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.hidden_dim_in_bytes = hidden_dim_in_bytes
        self.moe_in_dtype = moe_in_dtype
        self.moe_out_dtype = moe_out_dtype
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
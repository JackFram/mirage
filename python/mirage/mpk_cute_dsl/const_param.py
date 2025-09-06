import cutlass
import cutlass.cute as cute
import torch
import math
from typing import Optional, Type, Union
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils as utils

class ConstParam:
    def __init__(
            self, 
            # MoE param
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
            # mpk param
            mpk_queue_len: cutlass.Constexpr[int],
            num_worker_warps: cutlass.Constexpr[int],
            num_workers: cutlass.Constexpr[int],
            # gemm param
            thr_tile_shape: tuple[int, int],
            mma_tiler_mn: tuple[int, int],
            occupancy: cutlass.Constexpr[int],
            swapAB: bool,
            acc_dtype: cutlass.Constexpr[cutlass.Numeric],
            use_2cta_instrs: bool,
            cluster_shape_mn: tuple[int, int],
            tensormap_update_mode: cutlass.utils.TensorMapUpdateMode,
            # gemm kernel param
            tiled_mma: cute.TiledMma,
            w13_mA_mkl: cute.Tensor,
            w13_mB_nkl: cute.Tensor,
            w2_mA_mkl: cute.Tensor,
            w2_mB_nkl: cute.Tensor,
            w13_mC_mnl: cute.Tensor,
            cluster_layout_vmnk: cute.Layout,
            a_smem_layout_staged: cute.ComposedLayout,
            b_smem_layout_staged: cute.ComposedLayout,
            c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
            epi_tile: cute.Tile,
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

        # gemm const parameters
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = tensormap_update_mode
        self.delegate_tensormap_ab_init = (
            tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
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

        # Set barrier id for cta sync, epilog sync, tmem ptr sync and tensormap update sync
        self.cta_sync_bar_id = 2
        self.epilog_sync_bar_id = 3
        self.tmem_ptr_sync_bar_id = 4
        self.tensormap_ab_init_bar_id = 5
        self.num_tma_load_bytes = 0
        
        # gemm kernel const parameters
        self.tiled_mma = tiled_mma
        self.a_smem_layout_staged = a_smem_layout_staged
        self.b_smem_layout_staged = b_smem_layout_staged
        self.c_smem_layout_staged = c_smem_layout_staged
        self.w13_mA_mkl = w13_mA_mkl
        self.w13_mB_nkl = w13_mB_nkl
        self.w2_mA_mkl = w2_mA_mkl
        self.w2_mB_nkl = w2_mB_nkl
        self.w13_mC_mnl = w13_mC_mnl
        self.cluster_layout_vmnk = cluster_layout_vmnk
        self.epi_tile = epi_tile

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
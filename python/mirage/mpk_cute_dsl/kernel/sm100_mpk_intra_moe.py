from typing import List, Type, Union
from inspect import isclass

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.torch import dtype as torch_dtype

from mpk_cute_dsl.moe_utils import MoEParam
from mpk_cute_dsl.dist_utils import ProcessGroupInfo
from mpk_cute_dsl.kernel.gemm_utils import *
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_scheduler import MPKScheduler

from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam
from mpk_cute_dsl.const_param import ConstParam

"""
A persistent MoE kernel (dispatch+FFN+combine) with cute DSL on blackwell (SM100).
TODO(Zhihao): 
  1. Create a customized buffer class for moe communication buffer management
"""

class SM100MPKIntraMoEKernel:
    def __init__(
        self,
        moe_param: MoEParam,
        dist_param: ProcessGroupInfo,
        profiler_buffer_size: int = 1024,
        mpk_queue_len: int = 5120,
        profiler_enabled: bool = False,
    ):
        self.moe_param = moe_param
        self.dist_param = dist_param
        self.profiler_buffer_size = profiler_buffer_size
        self.profiler_enabled = profiler_enabled

        # Dist Info
        self.local_rank: cutlass.Constexpr[int] = dist_param.local_rank
        self.num_local_ranks: cutlass.Constexpr[int] = dist_param.world_local_size

        # Launching config
        self.num_warp: cutlass.Constexpr[int] = 9 # 1 warp for task fetching and rest for doing the actual work
        self.num_worker_warp: cutlass.Constexpr[int] = self.num_warp - 1 # 8 warps for doing the actual work
        self.threads_per_cta: cutlass.Constexpr[int] = 32 * self.num_warp
        self.smem_capacity = sm100_utils.SMEM_CAPACITY["sm100"]
        self.mpk_queue_len = mpk_queue_len
        
        # MoE meta info setup
        self.num_local_experts: cutlass.Constexpr[int] = int(moe_param.num_experts / (dist_param.world_size))
        self.hidden_dim: cutlass.Constexpr[int] = moe_param.hidden_dim
        # NOTE(Zhihao): assume the in_dtype and out_dtype are the same
        self.hidden_dim_in_bytes: cutlass.Constexpr[int] = moe_param.hidden_dim * torch_dtype(moe_param.in_dtype).itemsize
        self.num_tokens_per_rank: cutlass.Constexpr[int] = moe_param.num_tokens_per_rank
        self.max_num_tokens: cutlass.Constexpr[int] = self.num_tokens_per_rank * self.num_local_ranks
        self.combine_token_stride: cutlass.Constexpr[int] = cutlass.cute.round_up(self.hidden_dim_in_bytes, 16) # align to 16 bytes for 128b data transfer
        self.dispatch_token_stride: cutlass.Constexpr[int] = cutlass.cute.round_up(self.hidden_dim_in_bytes + 4, 16) # additional 4 bytes for meta data, align to 16 bytes for 128b data transfer
        if cutlass.const_expr(self.combine_token_stride % 16 != 0 or self.dispatch_token_stride % 16 != 0):
            raise TypeError(f"dispatch_token_stride {self.dispatch_token_stride} and combine_token_stride {self.combine_token_stride} should be divisible by 16")
        
        # Persistent kernel config
        sm_count = utils.HardwareInfo(torch.cuda.current_device()).get_device_multiprocessor_count()
        self.num_workers = sm_count # 148 for blackwell
        
        self.buffer_size_in_bytes: cutlass.Constexpr[int] = max(self.get_combine_buffer_size(), self.get_dispatch_buffer_size())
    
    def get_dispatch_buffer_size(self):
        size = 0

        size += 16
        size += cutlass.cute.round_up(self.num_local_experts * 4, 16)
        size += self.num_local_experts * self.num_tokens_per_rank * self.dispatch_token_stride

        return int(size)
    
    def get_combine_buffer_size(self):
        size = 0

        size += 16
        size += cutlass.cute.round_up(self.num_local_experts * 4, 16)
        size += self.num_local_experts * self.num_tokens_per_rank * self.combine_token_stride

        return int(size)    


    def _setup_attributes(self):
        # setting up attributes for dispatch and combine
        self.dispatch_buffer_offset_in_bytes: cutlass.Constexpr[int] = 0
        self.combine_buffer_offset_in_bytes: cutlass.Constexpr[int] = 4
        self.count_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16
        self.token_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16 + cutlass.cute.round_up(self.num_local_experts * 4, 16)
        # For dispatch and combine
        assert self.hidden_dim % (32 * self.num_worker_warp) == 0, "The hidden dimension should be divisible by the number of worker threads per CTA."
        self.thr_tile_shape = (1, self.hidden_dim//(32 * self.num_worker_warp))
        # Group GeMM attributes
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for Weight/Input
        - Computing epilogue subtile
        - Setting up Weight/Input/Output stage counts in shared memory
        - Computing Weight/Input/Output shared memory layout
        - Computing tensor memory allocation columns
        """

        if self.moe_param.swapAB:
            self.a_dtype = self.w_dtype
            self.b_dtype = self.i_dtype
            self.a_major_mode = self.w_major_mode
            self.b_major_mode = self.i_major_mode
            self.c_major_mode = self.o_layout.mma_major_mode()
            self.w13_a_tensor = self.kernel_param.w13_tensor
            self.w13_b_tensor = self.kernel_param.dispatch_recv_token_tensor
            self.w13_c_tensor = self.kernel_param.ffn_fused_w13_output_tensor
            self.w2_a_tensor = self.kernel_param.w2_tensor
            self.w2_b_tensor = self.kernel_param.ffn_fused_w13_output_tensor
        else:
            self.a_dtype = self.i_dtype
            self.b_dtype = self.w_dtype
            self.a_major_mode = self.i_major_mode
            self.b_major_mode = self.w_major_mode
            self.c_major_mode = self.o_layout.mma_major_mode()
            self.w13_a_tensor = self.kernel_param.dispatch_recv_token_tensor
            self.w13_b_tensor = self.kernel_param.w13_tensor
            self.w13_c_tensor = self.kernel_param.ffn_fused_w13_output_tensor
            self.w2_a_tensor = self.kernel_param.ffn_fused_w13_output_tensor
            self.w2_b_tensor = self.kernel_param.w2_tensor

        assert not self.moe_param.use_2cta_instrs, "we don't use 2cta instrs for MPK MoE currently"

        self.cta_group = tcgen05.CtaGroup.ONE

        self.mma_tiler = (*self.moe_param.mma_tiler_mn, 1)

        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.moe_param.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4 # make the cta tile size k to be 64 for bfloat16 due to the 128B swizzle requirement
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        
        if self.moe_param.swapAB:
            output_cta_tile_shape = (cta_tile_shape_mnk[1], cta_tile_shape_mnk[0], cta_tile_shape_mnk[2])
        else:
            output_cta_tile_shape = cta_tile_shape_mnk
        

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.moe_param.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        is_a_mcast = num_mcast_ctas_a > 1
        is_b_mcast = num_mcast_ctas_b > 1
        
        assert self.moe_param.use_tma_store, "we always use tma store currently"

        # Compute epilogue subtile
        if cutlass.const_expr(self.moe_param.use_tma_store):
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                cta_tile_shape_mnk,
                self.moe_param.use_2cta_instrs,
                self.o_layout,
                self.o_dtype,
            )
        else:
            self.epi_tile = self.cta_tile_shape_mnk[:2]
            
        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        used_smem_size = (18 + self.local_rank * self.num_local_experts) * 4 # used smem size for mpk task management and expert_send_count barrier
        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.o_dtype,
            self.o_layout,
            self.smem_capacity,
            used_smem_size,
            self.moe_param.occupancy,
            self.moe_param.use_tma_store,
        )
        
        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        # TODO(Zhihao): revisit here for swapAB
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.o_dtype,
                self.o_layout,
                self.epi_tile,
                self.num_c_stage,
            )
            if cutlass.const_expr(self.moe_param.use_tma_store)
            else None
        )

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )
    
    @cute.jit
    def __call__(
        self,
        kernel_param: MoEKernelParam,
        # profiler meta
        profiler_buffer: cute.Tensor,
        profiler_ptr: cute.Tensor,
        # cuda stream
        stream: cuda.CUstream,
    ):  
        # setup group gemm attributes

        self.w_dtype = kernel_param.w13_tensor.element_type
        self.i_dtype = kernel_param.dispatch_recv_token_tensor.element_type
        self.o_dtype = kernel_param.ffn_fused_w13_output_tensor.element_type
        self.w_major_mode = utils.LayoutEnum.from_tensor(kernel_param.w13_tensor[None, None, 0]).mma_major_mode()
        self.i_major_mode = utils.LayoutEnum.from_tensor(kernel_param.dispatch_recv_token_tensor[None, None, 0]).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(kernel_param.ffn_fused_w13_output_tensor[None, None, 0])

        self.kernel_param = kernel_param
        
        if cutlass.const_expr(self.w_dtype != self.i_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()
        
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.moe_param.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A 
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.moe_param.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        # Get TMA atom and tensor for A in fused_ffn_w13_task
        tma_atom_w13_a, tma_tensor_w13_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            self.w13_a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if self.w13_a_tensor.element_type is cutlass.Float32 else None
            ),
        )
        
        # Get TMA atom and tensor for A in fused_ffn_w2_task
        tma_atom_w2_a, tma_tensor_w2_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            self.w2_a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if self.w2_a_tensor.element_type is cutlass.Float32 else None
            ),
        )
        
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.moe_param.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        # Get TMA atom and tensor for B in fused_ffn_w13_task
        tma_atom_w13_b, tma_tensor_w13_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            self.w13_b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if self.w13_b_tensor.element_type is cutlass.Float32 else None
            ),
        )
        
        # Get TMA atom and tensor for B in fused_ffn_w2_task
        tma_atom_w2_b, tma_tensor_w2_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            self.w2_b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(
                cutlass.TFloat32 if self.w2_b_tensor.element_type is cutlass.Float32 else None
            ),
        )
        
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size
        
        # Setup TMA store for C in w13 task
        tma_atom_w13_c = None
        tma_tensor_w13_c = None
        if cutlass.const_expr(self.moe_param.use_tma_store):
            c_cta_v_layout = cute.composition(
                cute.make_identity_layout(self.w13_c_tensor.shape), self.epi_tile
            )
            epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
            tma_atom_w13_c, tma_tensor_w13_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                self.w13_c_tensor,
                epi_smem_layout,
                c_cta_v_layout,
            )

        self.buffer_align_bytes = 1024 # align to 1KB for smem swizzle requirement
        
        c_smem_size = (
            cute.cosize(self.c_smem_layout_staged.outer)
            if cutlass.const_expr(self.moe_param.use_tma_store)
            else 0
        )
        
        # Get launch parameters
        sm_count = utils.HardwareInfo(torch.cuda.current_device()).get_device_multiprocessor_count()
        grid_dim = [sm_count, 1, 1]
        block_dim = [self.threads_per_cta, 1, 1]
        
        @cute.struct
        class SharedStorage:
            send_index_buffer: cute.struct.MemRange[
                cutlass.Int32, 1
            ]
            mpk_task_sync_buffer: cute.struct.MemRange[
                cutlass.Int32, 1
            ]
            mpk_worker_sync_buffer: cute.struct.MemRange[
                cutlass.Int32, 16
            ]
            expert_send_count: cute.struct.MemRange[
                cutlass.Int32, self.num_local_experts * self.num_local_ranks
            ]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.o_dtype,
                    c_smem_size,
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        
        const_param = ConstParam(
            hidden_dim=self.hidden_dim,
            hidden_dim_in_bytes=self.hidden_dim_in_bytes,
            inter_dim=self.moe_param.inter_dim,
            moe_in_dtype=self.moe_param.in_dtype,
            num_topk=self.moe_param.num_topk,
            num_tokens_per_rank=self.num_tokens_per_rank,
            num_local_experts=self.num_local_experts,
            num_local_ranks=self.num_local_ranks,
            local_rank=self.local_rank,
            token_buffer_offset_in_bytes=self.token_buffer_offset_in_bytes,
            count_buffer_offset_in_bytes=self.count_buffer_offset_in_bytes,
            dispatch_token_stride=self.dispatch_token_stride,
            num_worker_warps=self.num_worker_warp,
            thr_tile_shape=self.thr_tile_shape,
            mma_tiler_mn=self.moe_param.mma_tiler_mn,
            swapAB=self.moe_param.swapAB,
            occupancy=self.moe_param.occupancy,
            mpk_queue_len=self.mpk_queue_len,
            num_workers=self.num_workers,
            acc_dtype=self.moe_param.acc_dtype,
            use_2cta_instrs=self.moe_param.use_2cta_instrs,
            cluster_shape_mn=self.moe_param.cluster_shape_mn,
            tensormap_update_mode=self.moe_param.tensormap_update_mode,
            tiled_mma=tiled_mma,
            w13_mA_mkl=tma_tensor_w13_a,
            w13_mB_nkl=tma_tensor_w13_b,
            w2_mA_mkl=tma_tensor_w2_a,
            w2_mB_nkl=tma_tensor_w2_b,
            w13_mC_mnl=tma_tensor_w13_c,
            cluster_layout_vmnk=self.cluster_layout_vmnk,
            a_smem_layout_staged=self.a_smem_layout_staged,
            b_smem_layout_staged=self.b_smem_layout_staged,
            c_smem_layout_staged=self.c_smem_layout_staged,
            epi_tile=self.epi_tile,
        )

        self.kernel(
            const_param=const_param,
            kernel_param=kernel_param,
            w13_tma_atom_a=tma_atom_w13_a,
            w13_tma_atom_b=tma_atom_w13_b,
            w13_tma_atom_c=tma_atom_w13_c,
            w2_tma_atom_a=tma_atom_w2_a,
            w2_tma_atom_b=tma_atom_w2_b,
            # profiler meta
            profiler_buffer=profiler_buffer,
            profiler_ptr=profiler_ptr,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        const_param: cutlass.Constexpr[ConstParam],
        kernel_param: MoEKernelParam,
        w13_tma_atom_a: cute.CopyAtom,
        w13_tma_atom_b: cute.CopyAtom,
        w13_tma_atom_c: cute.CopyAtom,
        w2_tma_atom_a: cute.CopyAtom,
        w2_tma_atom_b: cute.CopyAtom,
        # profiler meta
        profiler_buffer: cute.Tensor,
        profiler_ptr: cute.Tensor,
    ):

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        profiler = DslProfiler(
            profiler_buffer=profiler_buffer,
            profiler_ptr=profiler_ptr,
            buffer_size=self.profiler_buffer_size,
            profiler_enabled=self.profiler_enabled
        )
        
        scheduler = MPKScheduler(
            scheduler_warp_idx=self.num_warp-1, 
            smem_storage=storage,
            const_param=const_param,
            kernel_param=kernel_param,
            profiler=profiler,
        )
        
        # mega-kernel starts
        is_final_task = False
        while(cutlass.dynamic_expr(is_final_task == False)):
            scheduler.fetch_next_task()
            scheduler.sync_task()
            is_final_task = scheduler.execute_task(
                w13_tma_atom_a,
                w13_tma_atom_b,
                w13_tma_atom_c,
                w2_tma_atom_a,
                w2_tma_atom_b,
            )

    @cute.jit
    def combine_device(
        self,
        # input tensors
        rank_input_topk_indices: cute.Tensor,
        # output tensor
        rank_token_count: cute.Tensor,
        combine_send_token_tensor: cute.Tensor,
        output_tensor: cute.Tensor,
        # buffer ptr
        local_buffer_ptr: cute.Tensor,
        remote_buffer_ptr: cute.Tensor,
        count_buffer_ptr: cute.Tensor,
        # meta info tensors
        src_index: cute.Tensor, 
        src_expert: cute.Tensor,
        src_offset: cute.Tensor,
        src_rank: cute.Tensor,
    ):
        
        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        block_dim, _, _ = cute.arch.block_dim()
        grid_dim, _, _ = cute.arch.grid_dim()

        num_send_tokens = rank_token_count[0]

        # combine_send
        for rank_idx in range(block_idx * block_dim + thread_idx, self.num_local_ranks, grid_dim * block_dim):
            sync_tensor = self.get_combine_sync_buffer(remote_buffer_ptr, rank_idx)
            inline_ptx.st_flag_volatile(sync_tensor, cutlass.Uint32(1))
            count_sync_tensor = self.get_count_buffer_ptr(remote_buffer_ptr, rank_idx, 0)
            inline_ptx.st_flag_volatile(count_sync_tensor, cutlass.Uint32(1))

        for send_token_idx in range(block_idx, num_send_tokens, grid_dim):

            # TODO(Zhihao): use ld.global.nc (__ldg) to load the src_expert, src_index, src_offset, src_rank
            expert = src_expert[send_token_idx]
            index = src_index[send_token_idx]
            offset = src_offset[send_token_idx]
            rank = src_rank[send_token_idx]

            src_tensor = combine_send_token_tensor[(expert, offset, None, None)]
            dst_tensor = self.get_combine_token_ptr_buffer(
                remote_buffer_ptr,  
                rank,
                expert,
                index,
            )

            tiled_src_tensor = cute.zipped_divide(src_tensor, self.thr_tile_shape)
            thr_src_vec = tiled_src_tensor[(None, (0, thread_idx))]

            tiled_dst_tensor = cute.zipped_divide(dst_tensor, self.thr_tile_shape)
            thr_dst_vec = tiled_dst_tensor[(None, (0, thread_idx))]

            thr_dst_vec.store(thr_src_vec.load())

            cute.arch.sync_threads()

            if (thread_idx == 0):
                remote_count_tensor = self.get_all_gather_count_buffer_ptr(
                    count_buffer_ptr,
                    rank, 
                    index,
                )
                inline_ptx.add_flag_release(
                    remote_count_tensor,
                    cutlass.Uint32(1),
                )

        self.grid_sync()

        # combine_recv

        rank_token_count[0] = 0

        for recv_token_idx in range(block_idx, self.num_tokens_per_rank, grid_dim):
        
            if (thread_idx == 0):
                local_count_tensor = self.get_all_gather_count_buffer_ptr(
                    count_buffer_ptr,
                    self.local_rank,
                    recv_token_idx,
                )

                count = inline_ptx.ld_flag_acquire(local_count_tensor)
                while(cutlass.dynamic_expr(count != self.moe_param.num_topk)):
                    count = inline_ptx.ld_flag_acquire(local_count_tensor)

                local_count_tensor[0] = 0

            cute.arch.sync_threads()

            thr_tiled_output_tensor = cute.zipped_divide(output_tensor, self.thr_tile_shape)
            thr_dst_vec = thr_tiled_output_tensor[(None, (recv_token_idx, thread_idx))]

            acc_vec = None

            for idx in cutlass.range_constexpr(0, self.moe_param.num_topk, 1):
                expert = rank_input_topk_indices[recv_token_idx, idx]
                src_rank = expert // self.num_local_experts
                src_local_expert_idx = expert % self.num_local_experts

                # Get the token pointer from the remote buffer
                token_tensor = self.get_combine_token_ptr_buffer(
                    local_buffer_ptr,
                    src_rank,
                    src_local_expert_idx,
                    recv_token_idx,
                )

                tiled_token_tensor = cute.zipped_divide(token_tensor, self.thr_tile_shape)
                thr_src_vec = tiled_token_tensor[(None, (0, thread_idx))]

                if acc_vec is None:
                    acc_vec = thr_src_vec
                else:
                    acc_vec += thr_src_vec

            thr_dst_vec.store(acc_vec)

        for rank_idx in range(block_idx * block_dim + thread_idx, self.num_local_ranks, grid_dim * block_dim):
            count_sync_tensor = self.get_count_buffer_ptr(local_buffer_ptr, rank_idx, 0)
            value = inline_ptx.ld_flag_volatile(count_sync_tensor)
            while(cutlass.dynamic_expr(value != 1)):
                value = inline_ptx.ld_flag_volatile(count_sync_tensor)
            inline_ptx.st_flag_volatile(count_sync_tensor, cutlass.Uint32(0))  # reset the flag to 0

        self.grid_sync()

        for rank_idx in range(block_idx * block_dim + thread_idx, self.num_local_ranks, grid_dim * block_dim):
            sync_tensor = self.get_combine_sync_buffer(remote_buffer_ptr, rank_idx)
            inline_ptx.st_flag_volatile(sync_tensor, cutlass.Uint32(0))

    @cute.jit
    def make_global_tensor_from_buffer_ptr(
        self,
        dtype: Type[cutlass.Numeric],
        offset: cutlass.Int64,
        layout: cutlass.cute.typing.Layout,
        ptr_i64: cutlass.Int64,
    ):
        """
        Create a global tensor from a buffer pointer.
        Args:
            dtype (Type[cutlass.Numeric]): The data type of the tensor.
            offset (cutlass.Int64): The offset in bytes of the tensor in the buffer.
            layout (cutlass.cute.typing.Layout): The layout of the tensor.
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
        Returns:
            cute.Tensor: The global tensor.
        """
        if cutlass.const_expr(
            not isclass(dtype) or not issubclass(dtype, cutlass.Numeric)
        ):
            raise TypeError(
                f"dtype must be a type of cutlass.Numeric, got {type(dtype)}"
            )
        tensor_gmem_ptr = cute.make_ptr(
            dtype, ptr_i64+offset, cute.AddressSpace.gmem, assumed_align=16
        )
        tensor = cute.make_tensor(tensor_gmem_ptr, layout)
        return tensor
    

    def get_dispatch_sync_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
    ):
        """
        Get the dispatch sync buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The dispatch sync buffer.
        """
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Uint32,
                offset=self.dispatch_buffer_offset_in_bytes,
                layout=cute.make_layout((1,), stride=(1,)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_combine_sync_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
    ):
        """
        Get the dispatch sync buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The dispatch sync buffer.
        """
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Uint32,
                offset=self.combine_buffer_offset_in_bytes,
                layout=cute.make_layout((1,), stride=(1,)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_dispatch_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """

        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.dispatch_token_stride

        # cute.printf(">?? rank: {}, expert_idx: {}, recv_token_idx: {}, offset: {}", rank, expert_idx, recv_token_idx, offset)
        # cute.printf(">?? token_buffer_offset_in_bytes: {}", self.token_buffer_offset_in_bytes)

        return self.make_global_tensor_from_buffer_ptr(
                dtype=self.moe_param.in_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1, self.hidden_dim), stride=(self.hidden_dim, 1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )

    def get_dispatch_meta_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int32,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the meta pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The meta pointer buffer.
        """
        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.dispatch_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=ptr_offset + self.hidden_dim_in_bytes,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_combine_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64,
        recv_token_idx: cutlass.Int64,
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """
        ptr_offset = self.token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * self.max_num_tokens + recv_token_idx) * self.combine_token_stride

        return self.make_global_tensor_from_buffer_ptr(
                dtype=self.moe_param.out_dtype,
                offset=ptr_offset,
                layout=cute.make_layout((1,self.hidden_dim), stride=(self.hidden_dim,1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_count_buffer_ptr(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64 = 0,
    ):
        """
        Get the count buffer pointer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The count buffer pointer.
        """
        offset = self.count_buffer_offset_in_bytes + expert_idx * 4
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=offset,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )
    
    def get_all_gather_count_buffer_ptr(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        index: cutlass.Int64 = 0,
    ):
        """
        Get the all gather count buffer pointer from the buffer pointer.
        Args:
            buffer_ptr_tensor (cute.Tensor): Tensor of buffer pointers.
            rank (cutlass.Int32): The rank of the pointer.
            index (cutlass.Int64): The index of the count buffer to access.
        Returns:
            cute.Tensor: The all gather count buffer pointer.
        """
        offset = index * 4
        return self.make_global_tensor_from_buffer_ptr(
                dtype=cutlass.Int32,
                offset=offset,
                layout=cute.make_layout((1), stride=(1)),
                ptr_i64=buffer_ptr_tensor[rank],
            )

        
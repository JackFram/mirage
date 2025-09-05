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
            a_dtype = self.w_dtype
            b_dtype = self.i_dtype
            a_major_mode = self.w_major_mode
            b_major_mode = self.i_major_mode
            c_major_mode = self.o_layout.mma_major_mode()
        else:
            a_dtype = self.i_dtype
            b_dtype = self.w_dtype
            a_major_mode = self.i_major_mode
            b_major_mode = self.w_major_mode
            c_major_mode = self.o_layout.mma_major_mode()
            
        assert not self.moe_param.use_2cta_instrs, "we don't use 2cta instrs for MPK MoE currently"

        cta_group = (
            tcgen05.CtaGroup.TWO if self.moe_param.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        mma_tiler = (*self.moe_param.mma_tiler_mn, 1)

        # Configure tiled mma
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype,
            a_major_mode,
            b_major_mode,
            self.moe_param.acc_dtype,
            cta_group,
            mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4 # make the cta tile size k to be 64 for bfloat16 due to the 128B swizzle requirement
        mma_tiler = (
            mma_tiler[0],
            mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        cta_tile_shape_mnk = (
            mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            mma_tiler[1],
            mma_tiler[2],
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.moe_param.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        is_a_mcast = num_mcast_ctas_a > 1
        is_b_mcast = num_mcast_ctas_b > 1

        # Compute epilogue subtile
        if cutlass.const_expr(self.moe_param.use_tma_store):
            epi_tile = sm100_utils.compute_epilogue_tile_shape(
                cta_tile_shape_mnk,
                self.moe_param.use_2cta_instrs,
                self.o_layout,
                self.o_dtype,
            )
        else:
            epi_tile = self.cta_tile_shape_mnk[:2]
            
        # Setup A/B/C stage count in shared memory and ACC stage count in tensor memory
        used_smem_size = (18 + self.local_rank * self.num_local_experts) * 4 # used smem size for mpk task management and expert_send_count barrier
        num_acc_stage, num_ab_stage, num_c_stage = compute_stages(
            tiled_mma,
            mma_tiler,
            a_dtype,
            b_dtype,
            epi_tile,
            self.o_dtype,
            self.o_layout,
            self.smem_capacity,
            used_smem_size,
            self.occupancy,
            self.moe_param.use_tma_store,
        )
        
        # Compute A/B/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler,
            a_dtype,
            num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler,
            b_dtype,
            num_ab_stage,
        )
        self.c_smem_layout_staged = (
            sm100_utils.make_smem_layout_epi(
                self.o_dtype,
                self.o_layout,
                epi_tile,
                num_c_stage,
            )
            if cutlass.const_expr(self.moe_param.use_tma_store)
            else None
        )
        
        epilog_shape = cute.product_each(
            cute.shape(epi_tile)
        )
        print(epilog_shape)
        exit(0)

        # Compute the number of tensor memory allocation columns
        self.num_tmem_alloc_cols = compute_num_tmem_alloc_cols(
            tiled_mma, mma_tiler, num_acc_stage
        )
        
        self.const_param = ConstParam(
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
            occupancy=self.occupancy,
            mpk_queue_len=self.mpk_queue_len,
            num_workers=self.num_workers,
            acc_dtype=self.moe_param.acc_dtype,
            use_2cta_instrs=self.moe_param.use_2cta_instrs,
            cluster_shape_mn=self.moe_param.cluster_shape_mn,
            tensormap_update_mode=self.moe_param.tensormap_update_mode,
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
        self.o_dtype = kernel_param.combine_send_token_tensor.element_type
        self.w_major_mode = utils.LayoutEnum.from_tensor(kernel_param.w13_tensor[0, None, None]).mma_major_mode()
        self.i_major_mode = utils.LayoutEnum.from_tensor(kernel_param.dispatch_recv_token_tensor[0, None, None]).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(kernel_param.combine_send_token_tensor[0, None, None])
        
        self.occupancy = self.moe_param.occupancy
        
        if cutlass.const_expr(self.w_dtype != self.i_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        
        # TODO(Zhihao): continue implementing this part
        
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

        self.shared_storage = SharedStorage

        assert self.hidden_dim % (32 * self.num_worker_warp) == 0, "The hidden dimension should be divisible by the number of worker threads per CTA."

        self._setup_attributes()
        
        # # TODO(Zhihao): complete gemm initialization then remove this line
        # exit(0)

        self.kernel(
            kernel_param=kernel_param,
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
        kernel_param: MoEKernelParam,
        # profiler meta
        profiler_buffer: cute.Tensor,
        profiler_ptr: cute.Tensor,
    ):

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # self.send_index_buffer = storage.send_index_buffer.get_tensor(
        #     cute.make_layout((1), stride=(1))
        # )
        
        profiler = DslProfiler(
            profiler_buffer=profiler_buffer,
            profiler_ptr=profiler_ptr,
            buffer_size=self.profiler_buffer_size,
            profiler_enabled=self.profiler_enabled
        )
        
        scheduler = MPKScheduler(
            scheduler_warp_idx=self.num_warp-1, 
            smem_storage=storage,
            const_param=self.const_param,
            kernel_param=kernel_param,
            profiler=profiler,
        )

        # thread_idx, _, _ = cute.arch.thread_idx()
        # block_idx, _, _ = cute.arch.block_idx()
        
        # mega-kernel starts
        is_final_task = False
        while(cutlass.dynamic_expr(is_final_task == False)):
            scheduler.fetch_next_task()
            scheduler.sync_task()
            is_final_task = scheduler.execute_task()

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

        
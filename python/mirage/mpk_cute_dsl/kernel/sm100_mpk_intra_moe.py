import functools
from typing import List, Type, Union
from inspect import isclass

import math
import torch
import cuda.bindings.driver as cuda
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.torch import dtype as torch_dtype

from mpk_cute_dsl.moe_utils import MoEParam
from mpk_cute_dsl.dist_utils import ProcessGroupInfo
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

from mpk_cute_dsl.kernel.mpk_task_kernel.mpk_task import MPKScheduler
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler

"""
A persistent MoE kernel (dispatch+FFN+combine) with cute DSL on blackwell (SM100).
TODO(Zhihao): 
  1. 
"""

class SM100MPKIntraMoEKernel:
    def __init__(
        self,
        moe_param: MoEParam,
        dist_param: ProcessGroupInfo,
        profiler_buffer_size: int = 1024,
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

        # FFN attributes
        # Set specialized warp ids
        self.acc_dtype: Type[cutlass.Numeric] = moe_param.acc_dtype
        self.use_2cta_instrs = moe_param.use_2cta_instrs
        self.cluster_shape_mn = moe_param.cluster_shape_mn
        # K dimension is deferred in _setup_attributes
        self.mma_tiler = (*moe_param.mma_tiler_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if moe_param.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        self.tensormap_update_mode = moe_param.tensormap_update_mode
        # Delegate tensormap ab initialization to MMA warp when SMEM mode is used for better latency hiding
        self.delegate_tensormap_ab_init = (
            moe_param.tensormap_update_mode == utils.TensorMapUpdateMode.SMEM
        )

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False

        self.occupancy = 1

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
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        # Barrier ID used by MMA/TMA warps to signal A/B tensormap initialization completion
        self.tensormap_ab_init_bar_id = 4
        self.num_tma_load_bytes = 0
    
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
        self.dispatch_buffer_offset_in_bytes: cutlass.Constexpr[int] = 0
        self.combine_buffer_offset_in_bytes: cutlass.Constexpr[int] = 4
        self.count_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16
        self.token_buffer_offset_in_bytes: cutlass.Constexpr[int] = 16 + cutlass.cute.round_up(self.num_local_experts * 4, 16)

        self.thr_tile_shape = (1, self.hidden_dim//self.threads_per_cta)
    
    @cute.jit
    def __call__(
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
        # mpk meta
        mpk_task_queue: cute.Tensor,
        mpk_task_consume_idx: cute.Tensor,
        mpk_task_produce_idx: cute.Tensor,
        mpk_task_barrier: cute.Tensor,
        # profiler meta
        profiler_buffer: cute.Tensor,
        profiler_ptr: cute.Tensor,
        # cuda stream
        stream: cuda.CUstream,
    ):  
        # setup group gemm attributes

        self.a_dtype = self.moe_param.in_dtype
        self.b_dtype = self.moe_param.in_dtype
        self.c_dtype = self.moe_param.out_dtype
        self.a_major_mode = tcgen05.OperandMajorMode.K
        self.b_major_mode = tcgen05.OperandMajorMode.K
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        
        # Get launch parameters
        sm_count = utils.HardwareInfo(torch.cuda.current_device()).get_device_multiprocessor_count()
        grid_dim = [sm_count, 1, 1]
        block_dim = [self.threads_per_cta, 1, 1]
        smem_size = 196 * 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            send_index_buffer: cute.struct.MemRange[
                cutlass.Int32, 1
            ]
            mpk_task_sync_buffer: cute.struct.MemRange[
                cutlass.Int32, 1
            ]
            # tensormap_buffer: cute.struct.MemRange[
            #     cutlass.Int64, self.size_tensormap_in_i64
            # ]
            # ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            # ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            # acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            # acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            # tmem_dealloc_mbar_ptr: cutlass.Int64
            # tmem_holding_buf: cutlass.Int32
            # # (EPI_TILE_M, EPI_TILE_N, STAGE)
            # sC: cute.struct.Align[
            #     cute.struct.MemRange[
            #         self.c_dtype,
            #         cute.cosize(self.epi_smem_layout_staged.outer),
            #     ],
            #     self.buffer_align_bytes,
            # ]
            # # (MMA, MMA_M, MMA_K, STAGE)
            # sA: cute.struct.Align[
            #     cute.struct.MemRange[
            #         self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
            #     ],
            #     self.buffer_align_bytes,
            # ]
            # # (MMA, MMA_N, MMA_K, STAGE)
            # sB: cute.struct.Align[
            #     cute.struct.MemRange[
            #         self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
            #     ],
            #     self.buffer_align_bytes,
            # ]
        self.shared_storage = SharedStorage
        
        # You are here: finish impl

        assert self.hidden_dim % (32 * self.num_worker_warp) == 0, "The hidden dimension should be divisible by the number of worker threads per CTA."

        self._setup_attributes()
        
        self.kernel(
            rank_input_tensor=rank_input_tensor,
            rank_input_topk_indices=rank_input_topk_indices,
            num_tokens_per_local_expert_recv=num_tokens_per_local_expert_recv,
            local_token_send_count_per_expert=local_token_send_count_per_expert,
            rank_token_count=rank_token_count,
            dispatch_recv_token_tensor=dispatch_recv_token_tensor,
            combine_send_token_tensor=combine_send_token_tensor,
            output_tensor=output_tensor,
            local_buffer_ptr=local_buffer_ptr,
            remote_buffer_ptr=remote_buffer_ptr,
            count_buffer_ptr=count_buffer_ptr,
            recv_num_token_per_rank=recv_num_token_per_rank,
            src_index=src_index,
            src_expert=src_expert,
            src_offset=src_offset,
            src_rank=src_rank,
            src_token=src_token,
            # mpk meta
            mpk_task_queue=mpk_task_queue,
            mpk_task_consume_idx=mpk_task_consume_idx,
            mpk_task_produce_idx=mpk_task_produce_idx,
            mpk_task_barrier=mpk_task_barrier,
            # profiler meta
            profiler_buffer=profiler_buffer,
            profiler_ptr=profiler_ptr,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            smem=smem_size,
            stream=stream,
        )

    # GPU device kernel
    @cute.kernel
    def kernel(
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
        # mpk meta
        mpk_task_queue: cute.Tensor,
        mpk_task_consume_idx: cute.Tensor,
        mpk_task_produce_idx: cute.Tensor,
        mpk_task_barrier: cute.Tensor,
        # profiler meta
        profiler_buffer: cute.Tensor,
        profiler_ptr: cute.Tensor,
    ):

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        self.send_index_buffer = storage.send_index_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )
        mpk_task_sync_buffer = storage.mpk_task_sync_buffer.get_tensor(
            cute.make_layout((1), stride=(1))
        )
        
        profiler = DslProfiler(
            profiler_buffer=profiler_buffer,
            profiler_ptr=profiler_ptr,
            buffer_size=self.profiler_buffer_size,
            profiler_enabled=self.profiler_enabled
        )

        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        scheduler = MPKScheduler(
            scheduler_warp_idx=self.num_warp-1, 
            task_queue=mpk_task_queue,
            task_consume_idx=mpk_task_consume_idx,
            task_produce_idx=mpk_task_produce_idx,
            task_barrier=mpk_task_barrier,
            task_sync_buffer=mpk_task_sync_buffer,
            profiler=profiler,
        )
           
        # mega-kernel debug    
        scheduler.fetch_next_task()
        # if thread_idx % 32 == 0:
        #     cute.printf("block-{}, warp-{}, thread-{}", block_idx, warp_idx, thread_idx)
        scheduler.sync_task()
        scheduler.execute_task()

        # # mega-kernel starts
        # test_num_work = 1
        # while(test_num_work > 0):
        #     scheduler.fetch_next_task()
        #     scheduler.sync_task()
        #     scheduler.execute_task()
        #     test_num_work -= 1

    @cute.jit
    def dispatch_device(
        self,
        # input tensors
        rank_input_tensor: cute.Tensor,
        rank_input_topk_indices: cute.Tensor,
        # output tensor
        num_tokens_per_local_expert_recv: cute.Tensor,
        local_token_send_count_per_expert: cute.Tensor,
        rank_token_count: cute.Tensor,
        dispatch_recv_token_tensor: cute.Tensor,
        # buffer ptr
        local_buffer_ptr: cute.Tensor,
        remote_buffer_ptr: cute.Tensor,
        # meta info tensors
        recv_num_token_per_rank: cute.Tensor,
        src_index: cute.Tensor,
        src_expert: cute.Tensor,
        src_offset: cute.Tensor,
        src_rank: cute.Tensor,
        src_token: cute.Tensor,
    ):
        # dispatch send

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        block_dim, _, _ = cute.arch.block_dim()
        num_sm, _, _ = cute.arch.grid_dim()

        for rank_idx in range(block_idx * block_dim + thread_idx, self.num_local_ranks, num_sm * block_dim):
            sync_tensor = self.get_dispatch_sync_buffer(remote_buffer_ptr, rank_idx)
            inline_ptx.st_flag_volatile(sync_tensor, cutlass.Uint32(1))  # set the flag to 1 to indicate the dispatch starts

        for token_idx in range(block_idx, self.num_tokens_per_rank, num_sm):

            thr_tiled_rank_input_tensor = cute.zipped_divide(rank_input_tensor, self.thr_tile_shape)
            thr_src_vec = thr_tiled_rank_input_tensor[(None, (token_idx, thread_idx))]

            for topk_idx in cutlass.range_constexpr(0, self.moe_param.num_topk, 1):

                # Get the local expert index
                expert_idx = rank_input_topk_indices[token_idx, topk_idx]
                
                # Get the synchronized index for sending tokens
                if (thread_idx == 0):
                    recv_index = inline_ptx.atomic_add(local_token_send_count_per_expert[expert_idx, None], 1)
                    self.send_index_buffer[0] = recv_index
                cute.arch.sync_threads()
                remote_index = self.send_index_buffer[0]

                remote_rank = expert_idx // self.num_local_experts
                remote_expert_idx = expert_idx % self.num_local_experts

                remote_tensor = self.get_dispatch_token_ptr_buffer(
                    remote_buffer_ptr,
                    remote_rank,
                    remote_expert_idx,
                    remote_index,
                )

                meta_tensor = self.get_dispatch_meta_ptr_buffer(
                    remote_buffer_ptr,
                    remote_rank,
                    remote_expert_idx,
                    remote_index,
                )

                if (thread_idx == 0):
                    # Store the meta data
                    meta_tensor[0] = cutlass.Int32(token_idx)  # token index


                thr_tiled_rank_recv_tensor = cute.zipped_divide(remote_tensor, self.thr_tile_shape)
                thr_dst_vec = thr_tiled_rank_recv_tensor[(None, (0, thread_idx))]
                    
                thr_dst_vec.store(thr_src_vec.load())

                cute.arch.sync_threads()

        # grid_sync

        self.grid_sync()

        # send token count to remote buffer

        for expert_idx in range(block_idx * block_dim + thread_idx, self.moe_param.num_experts, num_sm * block_dim):
            remote_rank = expert_idx // self.num_local_experts
            remote_expert_idx = expert_idx % self.num_local_experts
            sync_tensor = self.get_count_buffer_ptr(remote_buffer_ptr, remote_rank, remote_expert_idx)
            inline_ptx.st_flag_release(sync_tensor, local_token_send_count_per_expert[expert_idx, 0] + 1)  # use the token count as the flag to indicate the dispatch send is done
            # cute.printf(">??-[send-{}] remote_rank: {}, remote_expert_idx: {}, token_count: {}", self.dist_param.local_rank, remote_rank, remote_expert_idx, local_token_send_count_per_expert[expert_idx])

        # dispatch recv

        # 1. use ld_acquire to wait for the token to be sent and collect meta info
        for recv_group_idx in range(block_idx, self.num_local_experts * self.num_local_ranks, num_sm):

            local_rank = recv_group_idx // self.num_local_experts
            local_expert_idx = recv_group_idx % self.num_local_experts
            
            if (thread_idx == 0): 
                count_tensor = self.get_count_buffer_ptr(local_buffer_ptr, local_rank, local_expert_idx)
                token_count = 0
                while(cutlass.dynamic_expr(token_count == 0)):
                    token_count = inline_ptx.ld_flag_acquire(count_tensor)
                inline_ptx.st_flag_release(count_tensor, cutlass.Int32(0))  # reset the flag to 0
                token_count -= 1

                recv_num_token_per_rank[recv_group_idx] = token_count

                self.block_expert_start_index[recv_group_idx] = inline_ptx.atomic_add(
                    num_tokens_per_local_expert_recv[local_expert_idx, None],
                    token_count,
                )

                self.block_token_start_index[recv_group_idx] = inline_ptx.atomic_add(
                    rank_token_count,
                    token_count,
                )

            cute.arch.sync_threads()

            token_count = recv_num_token_per_rank[recv_group_idx]
            expert_start = self.block_expert_start_index[recv_group_idx]
            token_start = self.block_token_start_index[recv_group_idx]

            for group_token_idx in range(thread_idx, token_count, block_dim):
            # if (thread_idx <  token_count):
                meta_tensor = self.get_dispatch_meta_ptr_buffer(
                    local_buffer_ptr,
                    local_rank,
                    local_expert_idx,
                    group_token_idx,
                )

                token_idx = token_start + group_token_idx  # absolute index of the token in the rank
                src_expert[token_idx] = local_expert_idx # relative expert index in the local rank
                src_offset[token_idx] = expert_start + group_token_idx # relative token index in the local expert
                src_rank[token_idx] = local_rank # relative rank index in the local world
                src_token[token_idx] = group_token_idx # relative token index in the local group ([local_rank, local_expert_idx])
                src_index[token_idx] = meta_tensor[0]

        self.grid_sync()

        # if(block_idx == 0 and thread_idx == 0):
        #     cute.printf(">??-[recv-{}] rank_token_count: {}", self.dist_param.local_rank, rank_token_count[0])

        # 2. cp from local buffer to output tensor (token parallel)
        for recv_token_idx in range(block_idx, rank_token_count[0], num_sm):
            dst_expert_offset = src_offset[recv_token_idx]
            dst_expert = src_expert[recv_token_idx]

            local_buffer_tensor = self.get_dispatch_token_ptr_buffer(
                    local_buffer_ptr,
                    src_rank[recv_token_idx],
                    dst_expert,
                    src_token[recv_token_idx],
                )
            tiled_src_tensor = cute.zipped_divide(local_buffer_tensor, self.thr_tile_shape)
            thr_src_vec = tiled_src_tensor[(None, (0, thread_idx))]

            dst_tensor = dispatch_recv_token_tensor[(dst_expert, dst_expert_offset, None, None)]
            tiled_dst_tensor = cute.zipped_divide(dst_tensor, self.thr_tile_shape)
            thr_dst_vec = tiled_dst_tensor[(None, (0, thread_idx))]
            thr_dst_vec.store(thr_src_vec.load())

        self.grid_sync()

        for rank_idx in range(block_idx * block_dim + thread_idx, self.num_local_ranks, num_sm * block_dim):
            sync_tensor = self.get_dispatch_sync_buffer(remote_buffer_ptr, rank_idx)
            inline_ptx.st_flag_volatile(sync_tensor, cutlass.Uint32(0))  # set the flag to 0 to indicate the dispatch finishes

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

        
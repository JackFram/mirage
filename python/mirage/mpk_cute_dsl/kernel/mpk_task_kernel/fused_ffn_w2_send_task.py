from operator import index
from typing import Optional, Union
import cutlass.cute as cute
import cutlass
import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx
import cutlass.utils.blackwell_helpers as sm100_utils

from typing import List, Type, Union
from inspect import isclass
from cutlass.cute.nvgpu import cpasync, tcgen05
from mpk_cute_dsl.kernel.mpk_task_kernel.undefined_task import UndefinedTask
from mpk_cute_dsl.profiler.dsl_profiler import DslProfiler
from mpk_cute_dsl.param import MoEKernelParam

from mpk_cute_dsl.const_param import ConstParam

from cutlass.cutlass_dsl import (
    new_from_mlir_values,
)
from cutlass._mlir import ir

class FusedFFNW2SendTask:
    def __init__(
            self, 
            task_desc: cutlass.Uint32,
            profiler: DslProfiler, 
            const_param: ConstParam, 
            kernel_param: MoEKernelParam, 
            smem_storage: cute.core.struct
        ):
        # Task Descripter Format:
        # | 31 - 28 |   23 - 16  |  15 - 8  |    7 - 0    |
        # | task_id | expert_idx | tile_idx | ffn_task_id |
        self.task_desc = task_desc
        self.profiler = profiler
        self.const_param = const_param
        self.kernel_param = kernel_param
        self.smem_storage = smem_storage
        self.task_name = "Fused-FFN-W2-Send-Task"

    def __extract_mlir_values__(self):
        values = self.task_desc.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "FusedFFNW2SendTask":
        assert len(values) == 1
        new_task_desc = new_from_mlir_values(
            self.task_desc, [values[0]]
        )
        return FusedFFNW2SendTask(new_task_desc, self.profiler, self.const_param, self.kernel_param, self.smem_storage)

    @cute.jit
    def execute(
        self,
        tiled_mma: cute.TiledMma,
        w2_tma_atom_a: cute.CopyAtom,
        w2_tma_atom_b: cute.CopyAtom,
        w2_mA_mkl: cute.Tensor,
        w2_mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        w2_d_tile: cute.Tile,
    ):
        # Execute the combine receive task
        self.profiler.profile_event(event_name="Fused-FFN-W2-Send", event_type="begin")
        self.fused_ffn_w2(
            tiled_mma=tiled_mma,
            w2_tma_atom_a=w2_tma_atom_a,
            w2_tma_atom_b=w2_tma_atom_b,
            w2_mA_mkl=w2_mA_mkl,
            w2_mB_nkl=w2_mB_nkl,
            cluster_layout_vmnk=cluster_layout_vmnk,
            a_smem_layout_staged=a_smem_layout_staged,
            b_smem_layout_staged=b_smem_layout_staged,
            c_smem_layout_staged=c_smem_layout_staged,
            epi_tile=epi_tile,
            w2_d_tile=w2_d_tile,
        )
        self.profiler.profile_event(event_name="Fused-FFN-W2-Send", event_type="end")

    @cute.jit
    def fused_ffn_w2(        
        self,
        tiled_mma: cute.TiledMma,
        w2_tma_atom_a: cute.CopyAtom,
        w2_tma_atom_b: cute.CopyAtom,
        w2_mA_mkl: cute.Tensor,
        w2_mB_nkl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        w2_d_tile: cute.Tile,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        expert_idx = (self.task_desc >> 16) & cutlass.Uint32(0x000000FF)
        tile_idx = (self.task_desc >> 8) & cutlass.Uint32(0x000000FF)
        ffn_w2_task_id = (self.task_desc) & cutlass.Uint32(0x000000FF)

        num_worker_warps = self.const_param.num_worker_warps
        swapAB = self.const_param.swapAB
        epilog_warp_id = self.const_param.epilog_warp_id
        tma_warp_id = self.const_param.tma_warp_id
        mma_warp_id = self.const_param.mma_warp_id
        num_acc_stage = self.const_param.num_acc_stage
        num_ab_stage = self.const_param.num_ab_stage
        num_c_stage = self.const_param.num_c_stage
        mma_tiler = self.const_param.mma_tiler
        d_mma_tiler = self.const_param.d_mma_tiler
        cta_sync_bar_id = self.const_param.cta_sync_bar_id
        epilog_sync_bar_id = self.const_param.epilog_sync_bar_id
        ffn_w2_k_cnt = self.const_param.ffn_w2_k_cnt
        num_tma_load_bytes = self.const_param.num_tma_load_bytes
        tmem_ptr_sync_bar_id = self.const_param.tmem_ptr_sync_bar_id
        c_dtype = self.const_param.c_dtype
        acc_dtype = self.const_param.acc_dtype
        cta_group = self.const_param.cta_group
        num_tmem_alloc_cols = self.const_param.num_tmem_alloc_cols
        token_tile_size = self.const_param.token_tile_size
        
        combine_info_tensor = self.kernel_param.combine_info_tensor
        remote_buffer_ptr = self.kernel_param.remote_buffer_ptr
        count_buffer_ptr = self.kernel_param.count_buffer_ptr

        # fused W1W3 pipelined accumulation + SwiGLU + element prod epilogue + SwapAB kernel 
        
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        
        # Prefetch tma desc
        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(w2_tma_atom_a)
            cpasync.prefetch_descriptor(w2_tma_atom_b)
        
        # no cluster launch and 2SM UMMA
        mma_tile_coord_v = 0
        is_leader_cta = True
        cta_rank_in_cluster = 0
        block_in_cluster_coord_vmnk = (0, 0, 0, 0)
        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #

        ab_full_mbar_ptr = self.smem_storage.ab_full_mbar_ptr.data_ptr()
        ab_empty_mbar_ptr = self.smem_storage.ab_empty_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = self.smem_storage.acc_full_mbar_ptr.data_ptr()
        acc_empty_mbar_ptr = self.smem_storage.acc_empty_mbar_ptr.data_ptr()
        tmem_holding_buf = self.smem_storage.tmem_holding_buf
        
        #  init barrier for loading A, B with TMA
        if warp_idx == epilog_warp_id[0]:
            for k_stage in cutlass.range_constexpr(num_ab_stage):
                num_tma_producer = 1
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(ab_full_mbar_ptr + k_stage, 1)
                    cute.arch.mbarrier_init(
                        ab_empty_mbar_ptr + k_stage, num_tma_producer
                    )
        # Accumulator barrier init
        if warp_idx == mma_warp_id:
            for acc_stage in cutlass.range_constexpr(num_acc_stage):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_init(acc_full_mbar_ptr + acc_stage, 1)
                    cute.arch.mbarrier_init(
                        acc_empty_mbar_ptr + acc_stage, 4
                    )
        cute.arch.mbarrier_init_fence()
        sC_send_layout = cute.make_layout((d_mma_tiler[1], w2_d_tile[0].shape, num_c_stage), stride=(w2_d_tile[0].shape, 1, d_mma_tiler[1] * w2_d_tile[0].shape))
        #
        # Setup smem tensor A/B/C
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = self.smem_storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sC_send = self.smem_storage.sC.get_tensor(
            sC_send_layout, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = self.smem_storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = self.smem_storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        
        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            w2_mA_mkl, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            w2_mB_nkl, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
        )
        
        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)

        #
        # Partition global/shared tensor for load A, B with TMA
        #
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            w2_tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            w2_tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        
        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, num_acc_stage)
        )
        
        cute.arch.barrier(
            barrier_id=cta_sync_bar_id, number_of_threads=num_worker_warps * 32
        )

        #
        # Specialized TMA load warp
        #
        if warp_idx == tma_warp_id:
            if cutlass.const_expr(swapAB):
                mma_tile_coord_mnl = (ffn_w2_task_id, tile_idx, expert_idx)
            else:
                mma_tile_coord_mnl = (tile_idx, ffn_w2_task_id, expert_idx)
            #
            # Slice to per mma tile index
            #
            # ((atom_v, rest_v), RestK)
            tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
            # ((atom_v, rest_v), RestK)
            tBgB_slice = tBgB[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            
            tma_wr_k_block = cutlass.Int32(0)
            smem_wr_buffer = tma_wr_k_block % num_ab_stage
            tma_wr_ab_empty_phase = tma_wr_k_block // num_ab_stage % 2 ^ 1
            peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                tma_wr_k_block < ffn_w2_k_cnt,
                ab_empty_mbar_ptr + smem_wr_buffer,
                tma_wr_ab_empty_phase,
            )
            

            for k_block in cutlass.range(0, ffn_w2_k_cnt, 1, unroll=1):
                tma_wr_k_block_next = tma_wr_k_block + 1
                smem_wr_buffer_next = tma_wr_k_block_next % num_ab_stage
                tma_wr_ab_empty_phase_next = (
                    tma_wr_ab_empty_phase ^ 1
                    if smem_wr_buffer_next == 0
                    else tma_wr_ab_empty_phase
                )
                
                smem_full_mbar_ptr = ab_full_mbar_ptr + smem_wr_buffer
                
                # Wait for AB buffer empty
                if peek_ab_empty_status == 0:
                    cute.arch.mbarrier_wait(
                        ab_empty_mbar_ptr + smem_wr_buffer, tma_wr_ab_empty_phase
                    )
                    
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        smem_full_mbar_ptr, num_tma_load_bytes
                    )
                
                # Load A/B with TMA
                cute.copy(
                    w2_tma_atom_a,
                    tAgA_slice[(None, tma_wr_k_block)],
                    tAsA[(None, smem_wr_buffer)],
                    tma_bar_ptr=smem_full_mbar_ptr,
                    mcast_mask=None,
                )
                cute.copy(
                    w2_tma_atom_b,
                    tBgB_slice[(None, tma_wr_k_block)],
                    tBsB[(None, smem_wr_buffer)],
                    tma_bar_ptr=smem_full_mbar_ptr,
                    mcast_mask=None,
                )

                # Peek (try_wait) AB buffer empty for k_block = prefetch_k_block_cnt + k_block + 1
                peek_ab_empty_status = cute.arch.mbarrier_conditional_try_wait(
                    tma_wr_k_block_next < ffn_w2_k_cnt,
                    ab_empty_mbar_ptr + smem_wr_buffer_next,
                    tma_wr_ab_empty_phase_next,
                )

                tma_wr_k_block = tma_wr_k_block_next
                smem_wr_buffer = smem_wr_buffer_next
                tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next
                
        #
        # Specialized MMA warp
        #
        if warp_idx == mma_warp_id:
            #  Bar sync for retrieve tmem ptr from shared mem
            tmem_ptr_read_threads = 32 * len((mma_warp_id, *epilog_warp_id))
            cute.arch.barrier(
                barrier_id=tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_buf_idx = 0
            # (MMA, MMA_M, MMA_N)
            tCtAcc = tCtAcc_base[(None, None, None, acc_buf_idx)]
            # Peek (try_wait) AB buffer full for k_block = 0
            mma_rd_k_block = cutlass.Int32(0)
            smem_rd_buffer = mma_rd_k_block % num_ab_stage
            need_check_rd_buffer_full = mma_rd_k_block < ffn_w2_k_cnt
            mma_rd_ab_full_phase = mma_rd_k_block // num_ab_stage % 2

            peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                need_check_rd_buffer_full,
                ab_full_mbar_ptr + smem_rd_buffer,
                mma_rd_ab_full_phase,
            )
            
            acc_empty_phase = 1 # no task fusion so always 1 (num_tiles_executed // num_acc_stage % 2 ^ 1)
            cute.arch.mbarrier_wait(
                acc_empty_mbar_ptr + acc_buf_idx, acc_empty_phase
            )
            
            #
            # Reset the ACCUMULATE field
            #
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            
            #
            # Mma mainloop
            #
            for k_block in cutlass.range_constexpr(ffn_w2_k_cnt, unroll_full=True):
                mma_rd_k_block_next = cutlass.Int32(k_block + 1)
                smem_rd_buffer_next = mma_rd_k_block_next % num_ab_stage
                mma_rd_ab_full_phase_next = (
                    mma_rd_ab_full_phase ^ 1
                    if smem_rd_buffer_next == 0
                    else mma_rd_ab_full_phase
                )
                # Wait for AB buffer full
                if peek_ab_full_status == 0:
                    cute.arch.mbarrier_wait(
                        ab_full_mbar_ptr + smem_rd_buffer, mma_rd_ab_full_phase
                    )
                # tCtAcc += tCrA * tCrB
                num_kphases = cute.size(tCrA, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx, smem_rd_buffer)

                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[kphase_coord],
                        tCrB[kphase_coord],
                        tCtAcc,
                    )
                    # Enable accumulate on tCtAcc after first kphase
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    
                # Async arrive AB buffer empty
                with cute.arch.elect_one():
                    tcgen05.commit(
                        ab_empty_mbar_ptr + smem_rd_buffer,
                        None,  # ab_empty_mcast_mask
                        cta_group,
                    )

                # Peek (try_wait) AB buffer full for k_block = k_block + 1
                need_check_rd_buffer_full = (
                    mma_rd_k_block_next < ffn_w2_k_cnt
                )

                peek_ab_full_status = cute.arch.mbarrier_conditional_try_wait(
                    need_check_rd_buffer_full,
                    ab_full_mbar_ptr + smem_rd_buffer_next,
                    mma_rd_ab_full_phase_next,
                )

                mma_rd_k_block = mma_rd_k_block_next
                smem_rd_buffer = smem_rd_buffer_next
                mma_rd_ab_full_phase = mma_rd_ab_full_phase_next
                
            with cute.arch.elect_one():
                tcgen05.commit(
                    acc_full_mbar_ptr + acc_buf_idx,
                    None,  # acc_full_mcast_mask
                    cta_group,
                )
                
        #
        # Specialized epilogue warps
        #
        if warp_idx < mma_warp_id:
            # Alloc tensor memory buffer
            if warp_idx == epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=False,
                )
            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem_ptr_read_threads = 32 * len((mma_warp_id, *epilog_warp_id))
            cute.arch.barrier(
                barrier_id=tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )
            
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_tidx = thread_idx
            #
            # Partition for epilogue
            #
            (
                tiled_copy_t2r,
                tTR_tAcc_base, 
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, epi_tile, False
            )
            
            #
            # Slice to per mma tile index
            #
            # Set tensor memory buffer for current tile
            acc_buf_idx = 0
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_buf_idx)]
            
            #
            # Wait for accumulator buffer full
            #
            acc_full_phase = 0
            cute.arch.mbarrier_wait(acc_full_mbar_ptr + acc_buf_idx, acc_full_phase)

            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc)) # (((32,32),1),1,1,(1,2))
            
            #
            # Store accumulator to global memory in subtiles
            #
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            
            for subtile_idx in range(subtile_cnt, unroll_full=True):
                #
                # Load accumulator from tensor memory buffer to register
                #
                # epilogue transpose + combine send for swapAB
                vec_stride = cute.size(tTR_rAcc)
                epilog_threads = 32 * len(epilog_warp_id)
                num_vec = epilog_threads // vec_stride
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                epi_buffer = subtile_idx % num_c_stage
                sC[None, thread_idx, epi_buffer].store(tTR_rAcc.load().to(c_dtype))
                # barrier to make sure shared memory store is visible to remote store
                cute.arch.barrier(
                    barrier_id=epilog_sync_bar_id,
                    number_of_threads=epilog_threads,
                )
                thr_src_vec = sC_send[thread_idx, None, epi_buffer]
                token_idx = thread_idx // num_vec + subtile_idx * vec_stride + tile_idx * token_tile_size
                packed_val = inline_ptx.ld_flag_relaxed_gpu_u32(combine_info_tensor[expert_idx, token_idx, None])
                valid_flag = packed_val >> 17 & cutlass.Uint32(0x00000001)
                src_rank_idx = packed_val >> 13 & cutlass.Uint32(0x0000000F)
                src_expert_idx = packed_val >> 8 & cutlass.Uint32(0x0000001F)
                src_token_idx = packed_val & cutlass.Uint32(0x000000FF)

                if valid_flag == 1:
                    thr_dst_vec = self.get_combine_token_ptr_buffer(
                        buffer_ptr_tensor=remote_buffer_ptr,
                        rank=src_rank_idx,
                        expert_idx=src_expert_idx,
                        recv_token_idx=src_token_idx,
                        ffn_tile_idx=ffn_w2_task_id,
                        vec_idx=thread_idx % num_vec,
                        tile_stride=epilog_threads,
                        vec_stride=vec_stride,
                    )
                    thr_dst_vec.store(thr_src_vec.load())
                
                cute.arch.barrier(
                    barrier_id=epilog_sync_bar_id,
                    number_of_threads=epilog_threads,
                )

                # atomic add release here. TODO(Zhihao): optimize this, overhead might be large
                if thread_idx % num_vec == 0 and valid_flag == 1:
                    remote_count_tensor = self.get_all_gather_count_buffer_ptr(
                        count_buffer_ptr,
                        src_rank_idx,
                        src_token_idx,
                    )
                    # TODO(Zhihao): might can use red instructions here
                    ret_count = inline_ptx.atomic_add_flag_release_sys_global_u32(
                        remote_count_tensor,
                        cutlass.Uint32(1),
                    )
                    
                    # if ret_count + 1 == 1024 * 40:
                    #     cute.printf("arrived!")
                        
                    # if ret_count + 1 > 1024 * 40:
                    #     cute.printf("overflow!")

            #
            # Async arrive accumulator buffer empty
            #
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(
                    acc_empty_mbar_ptr + acc_buf_idx,
                )
                
            #
            # Dealloc the tensor memory buffer
            #
            
            epilog_threads = 32 * len(epilog_warp_id)
            cute.arch.barrier(
                barrier_id=epilog_sync_bar_id, number_of_threads=epilog_threads
            )
            if warp_idx == epilog_warp_id[0]:
                cute.arch.dealloc_tmem(
                    tmem_ptr, num_tmem_alloc_cols, is_two_cta=False
                )

            #
            # Wait a/b buffer empty
            #
            if warp_idx == epilog_warp_id[0]:
                cute.arch.mbarrier_wait(
                    (ab_empty_mbar_ptr + ((ffn_w2_k_cnt - 1) % num_ab_stage)),
                    (((ffn_w2_k_cnt - 1) // num_ab_stage) % 2),
                )

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc: The accumulated tensor in register used to hold t2r results
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # get const param
        cta_tile_shape_mnk = self.const_param.cta_tile_shape_mnk
        c_layout = self.const_param.c_layout
        c_dtype = self.const_param.c_dtype
        acc_dtype = self.const_param.acc_dtype
        
        # Make tiledCopy for tensor memory load(t2r)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            cta_tile_shape_mnk,
            c_layout,
            c_dtype,
            acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)],
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_epi_C = thr_copy_t2r.partition_D(tAcc_epi)
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc = cute.make_fragment(
            tTR_epi_C[(None, None, None, 0, 0, 0)].shape, acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc
    
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
    
    def get_combine_token_ptr_buffer(
        self,
        buffer_ptr_tensor: cute.Tensor,
        rank: cutlass.Int32,
        expert_idx: cutlass.Int64,
        recv_token_idx: cutlass.Int64,
        ffn_tile_idx: cutlass.Int32,
        vec_idx: cutlass.Int32,
        tile_stride: cutlass.Constexpr[int],
        vec_stride: cutlass.Constexpr[int],
    ):
        """
        Get the token pointer buffer from the buffer pointer.
        Args:
            ptr_i64 (cutlass.Int64): The pointer to the buffer.
            rank (cutlass.Int32): The rank of the process.
        Returns:
            cute.Tensor: The token pointer buffer.
        """
        moe_out_dtype = self.const_param.moe_out_dtype
        num_tokens_per_rank = self.const_param.num_tokens_per_rank
        token_buffer_offset_in_bytes = self.const_param.token_buffer_offset_in_bytes
        combine_token_stride = self.const_param.combine_token_stride

        ptr_offset = token_buffer_offset_in_bytes
        ptr_offset += (expert_idx * num_tokens_per_rank + recv_token_idx) * combine_token_stride + (ffn_tile_idx * tile_stride + vec_idx * vec_stride) * (moe_out_dtype.width // 8)

        return self.make_global_tensor_from_buffer_ptr(
                dtype=moe_out_dtype,
                offset=ptr_offset,
                layout=cute.make_layout(vec_stride),
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
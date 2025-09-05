import dataclasses
import torch
import cutlass
import cutlass.utils as utils

@dataclasses.dataclass
class MoEParam:
    num_experts: int
    num_topk: int
    hidden_dim: int
    inter_dim: int
    num_tokens_per_rank: int
    in_dtype: torch.dtype = cutlass.Float16
    out_dtype: torch.dtype = cutlass.Float16
    # Grouped GEMM parameters
    acc_dtype: torch.dtype = cutlass.Float32
    use_2cta_instrs: bool = False
    use_tma_store: bool = True
    swapAB: bool = True
    mma_tiler_mn: tuple[int, int] = (128, 64)
    cluster_shape_mn: tuple[int, int] = (1, 1)
    tensormap_update_mode: utils.TensorMapUpdateMode = utils.TensorMapUpdateMode.SMEM
    occupancy: int = 1
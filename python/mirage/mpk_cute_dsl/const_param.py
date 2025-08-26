import cutlass
import cutlass.cute as cute

class ConstParam:
    def __init__(
            self, 
            num_topk: cutlass.Constexpr[int],
            num_worker_warps: cutlass.Constexpr[int],
            thr_tile_shape: tuple[int, int]
        ):
        
        # kernel const parameters
        self.num_worker_warps = num_worker_warps
        self.thr_tile_shape = thr_tile_shape

        # moe const parameters
        self.num_topk = num_topk
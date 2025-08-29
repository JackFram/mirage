import cutlass
import cutlass.cute as cute
from enum import Enum

class MPKTask(Enum):
    kFetch: cutlass.Uint32 = 0
    kHistAll2All: cutlass.Uint32 = 1
    kDispatchSend: cutlass.Uint32 = 2
    kDispatchRecv: cutlass.Uint32 = 3
    kFusedFFNW13: cutlass.Uint32 = 4
    kFusedFFNW2Send: cutlass.Uint32 = 5
    kCombineRecv: cutlass.Uint32 = 6
    kTerminate: cutlass.Uint32 = 7
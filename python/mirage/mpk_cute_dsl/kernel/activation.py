import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def silu(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(x) = x * sigmoid(x) = x * (1 + tanh(x / 2)) / 2 = (0.5 * x) * tanh(0.5 * x) + (0.5 * x)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * x, MUFU.TANH, and FFMA.
    """
    x_half = 0.5 * x
    return x_half * tanh(x_half) + x_half
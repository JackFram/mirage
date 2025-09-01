from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int, Boolean, Int32, Float32, Numeric, as_numeric, Uint32, Uint64
from typing import Optional, Tuple, Union, Callable


@dsl_user_op
def atomic_add(input: cute.Tensor, value: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Int32:
    """
    Perform an atomic addition on the input tensor using NVVM.
    This function assumes that the input tensor is a pointer to an integer type.
    """
    llvm_ptr = input.iterator.llvm_ptr
    res = nvvm.atomicrmw(res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=llvm_ptr, a=cutlass.Int32(value).ir_value())
    return res

@dsl_user_op
def st_flag_volatile(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_addr_ptr_i64, Uint32(flag).ir_value(loc=loc, ip=ip)],
        "st.volatile.global.u32 [$0], $1;",
        "l, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    
@dsl_user_op
def st_flag_volatile_64(sync_tensor: cute.Tensor, flag: Uint64, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_addr_ptr_i64, Uint64(flag).ir_value(loc=loc, ip=ip)],
        "st.volatile.global.u64 [$0], $1;",
        "l, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def ld_flag_volatile(sync_tensor: cute.Tensor, *, loc=None, ip=None) -> Uint32:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [flag_addr_ptr_i64],
            "ld.volatile.global.u32 $0, [$1];",
            "=r, l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    
@dsl_user_op
def ld_flag_relaxed_gpu_u32(sync_tensor: cute.Tensor, *, loc=None, ip=None) -> Uint32:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [flag_addr_ptr_i64],
            "ld.relaxed.gpu.u32 $0, [$1];",
            "=r, l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    
@dsl_user_op
def ld_flag_relaxed_sys_u32(sync_tensor: cute.Tensor, *, loc=None, ip=None) -> Uint32:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [flag_addr_ptr_i64],
            "ld.relaxed.sys.u32 $0, [$1];",
            "=r, l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def ld_flag_sys_acquire_u32(sync_tensor: cute.Tensor, *, loc=None, ip=None) -> Uint32:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [flag_addr_ptr_i64],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r, l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def st_flag_release(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [Uint32(flag).ir_value(loc=loc, ip=ip), flag_addr_ptr_i64],
        "st.release.sys.global.u32 [$1], $0;",
        "r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def st_flag_relaxed_sys_u32(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [Uint32(flag).ir_value(loc=loc, ip=ip), flag_addr_ptr_i64],
        "st.relaxed.sys.global.u32 [$1], $0;",
        "r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def st_flag_relaxed_gpu_u32(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [Uint32(flag).ir_value(loc=loc, ip=ip), flag_addr_ptr_i64],
        "st.relaxed.gpu.global.u32 [$1], $0;",
        "r, l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def add_flag_release(sync_tensor: cute.Tensor, value: Uint32, *, loc=None, ip=None) -> Uint32:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [flag_addr_ptr_i64, Uint32(value).ir_value(loc=loc, ip=ip)],
            "atom.release.sys.global.add.u32 $0, [$1], $2;",
            "=r, l, r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    
@dsl_user_op
def nanosleep(time: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Uint32(time).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    
@dsl_user_op
def get_globaltimer_lo(*, loc=None, ip=None) -> None:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %globaltimer_lo;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    
@dsl_user_op
def red_add_shared_u32(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_addr_ptr_i64, Uint32(flag).ir_value(loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "l, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    
@dsl_user_op
def red_add_global_u32(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_addr_ptr_i64, Uint32(flag).ir_value(loc=loc, ip=ip)],
        "red.global.add.u32 [$0], $1;",
        "l, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def red_add_global_release_u32(sync_tensor: cute.Tensor, flag: Uint32, *, loc=None, ip=None) -> None:
    flag_addr_ptr_i64 = sync_tensor.iterator.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [flag_addr_ptr_i64, Uint32(flag).ir_value(loc=loc, ip=ip)],
        "red.release.global.add.u32 [$0], $1;",
        "l, r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
import cutlass
import cutlass.cute as cute

@cute.struct
class SharedStorage:
    send_index_buffer: cute.struct.MemRange[
        cutlass.Int32, 1
    ]
    mpk_task_sync_buffer: cute.struct.MemRange[
        cutlass.Int32, 1
    ]
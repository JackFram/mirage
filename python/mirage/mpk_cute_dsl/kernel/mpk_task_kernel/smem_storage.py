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
    mpk_worker_sync_buffer: cute.struct.MemRange[
        cutlass.Int32, 16
    ]
    # TODO(Zhihao): hardcoded with number of expert, find a way to initialize from config
    expert_send_count: cute.struct.MemRange[
        cutlass.Int32, 32
    ]

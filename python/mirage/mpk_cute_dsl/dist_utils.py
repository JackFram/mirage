import dataclasses
import logging
import os
from collections.abc import Callable
from typing import Concatenate, ParamSpec, List, Optional, Tuple

import torch
from torch.multiprocessing import spawn  # pyright: ignore[reportPrivateImportUsage]
from mpk_cute_dsl.cuda_utils import checkCudaErrors
import cuda.bindings.runtime as cudart
import torch.distributed as dist
import ctypes

P = ParamSpec("P")

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProcessGroupInfo:
    world_size: int
    world_local_size: int
    rank: int
    node_rank: int
    local_rank: int
    device: torch.device


def _worker_parallel_launch(
    local_rank: int,
    world_size: int,
    world_local_size: int,
    node_rank: int,
    init_method: str,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    rank = node_rank * world_local_size + local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    world_group = torch.distributed.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)

    barrier = torch.tensor([rank], device=device)
    torch.distributed.all_reduce(barrier)

    try:
        worker(
            ProcessGroupInfo(
                world_size=world_size,
                world_local_size=world_local_size,
                rank=rank,
                node_rank=node_rank,
                local_rank=local_rank,
                device=device,
            ),
            *args,
            **kwargs,
        )
    except Exception:
        logger.exception("Error in worker function of parallel_launch")
        raise
    finally:
        torch.distributed.destroy_process_group()


def parallel_launch(
    world_size: int,
    worker: Callable[Concatenate[ProcessGroupInfo, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    assert not kwargs
    spawn(
        _worker_parallel_launch,
        args=(
            world_size,
            world_size,
            0,
            "tcp://localhost:29501",
            worker,
        )
        + args,
        nprocs=world_size,
        join=True,
    )

# def intranode_buffer_init_all_to_all(
#         buffer_size_in_bytes: int,
#         dist_param: ProcessGroupInfo,
# ):
#     local_buffer_ptr_list = []
#     ipc_handle_list = []
#     torch_buffer_list = []
#     for rank in range(dist_param.world_local_size):
#         torch_buffer = (
#             torch.empty(
#                 (
#                     buffer_size_in_bytes // 4,
#                 ),
#                 dtype=torch.int32,
#             )
#             .fill_(dist_param.local_rank * dist_param.world_local_size + rank)  # Fill with local rank for debugging
#             .cuda()
#         )
#         torch_buffer_list.append(torch_buffer)
#         local_buffer_ptr_list.append(torch_buffer.data_ptr())
#         ipc_handle = checkCudaErrors(cudart.cudaIpcGetMemHandle(torch_buffer.data_ptr()))
#         # print(f"Rank {dist_param.local_rank}: IPC handle for local buffer rank {rank}: {ipc_handle}")
#         ipc_handle_tensor = torch.tensor(bytearray(ipc_handle.reserved), dtype=torch.uint8).cuda()
#         ipc_handle_list.append(ipc_handle_tensor)

#     remote_ipc_handle_list = [torch.empty_like(ipc_handle_list[1]) for _ in range(dist_param.world_local_size)]
#     dist.all_to_all(
#         remote_ipc_handle_list,
#         ipc_handle_list,
#     )

#     handles = []
#     for tensor in remote_ipc_handle_list:
#         bytes_data = tensor.cpu().numpy().tobytes()
#         handle = cudart.cudaIpcMemHandle_t()
#         handle.reserved = bytes_data
#         handles.append(handle)
    
#     remote_buffer_ptr_list: List[int] = []
#     for i in range(dist_param.world_local_size):
#         if i == dist_param.local_rank:
#             remote_buffer_ptr_list.append(local_buffer_ptr_list[i])
#         elif checkCudaErrors(cudart.cudaDeviceCanAccessPeer(dist_param.local_rank, i)):
#             try:
#                 opened_ptr = checkCudaErrors(cudart.cudaIpcOpenMemHandle(
#                     handles[i], cudart.cudaIpcMemLazyEnablePeerAccess
#                 ))
#                 # print(f"Rank {dist_param.local_rank}: Opened IPC handle from rank {i}: {handles[i]}")
#                 remote_buffer_ptr_list.append(opened_ptr)
#             except Exception as e:
#                 print(f"Rank {dist_param.local_rank}: Failed to open IPC handle from rank {i}: {e}")
#                 raise
#         else:
#             raise RuntimeError(
#                 f"Rank {dist_param.local_rank} cannot access rank {i}'s memory. "
#                 "Ensure that all ranks are on the same node and can access each other's memory."
#             )
    
#     dist.barrier()
#     return torch_buffer_list, local_buffer_ptr_list, remote_buffer_ptr_list
    

# def intranode_buffer_init_all_gather(
#         buffer_size_in_bytes: int,
#         dist_param: ProcessGroupInfo,
# ):
#     torch_buffer = (
#             torch.empty(
#                 (
#                     buffer_size_in_bytes // 4,
#                 ),
#                 dtype=torch.int32,
#             )
#             .fill_(dist_param.local_rank)
#             .cuda()
#         )
#     tensor_ptr = torch_buffer.data_ptr()
#     ipc_handle = checkCudaErrors(cudart.cudaIpcGetMemHandle(tensor_ptr))
#     ipc_handle_tensor = torch.tensor(bytearray(ipc_handle.reserved), dtype=torch.uint8).cuda()

#     remote_ipc_handle_list = [torch.empty_like(ipc_handle_tensor) for _ in range(dist_param.world_local_size)]
#     dist.all_gather(
#         remote_ipc_handle_list,
#         ipc_handle_tensor,
#     )

#     handles = []
#     for tensor in remote_ipc_handle_list:
#         bytes_data = tensor.cpu().numpy().tobytes()
#         handle = cudart.cudaIpcMemHandle_t()
#         handle.reserved = bytes_data
#         handles.append(handle)

#     remote_buffer_ptr_list = []
#     for i in range(dist_param.world_local_size):
#         if i == dist_param.local_rank:
#             remote_buffer_ptr_list.append(tensor_ptr)
#         elif checkCudaErrors(cudart.cudaDeviceCanAccessPeer(dist_param.local_rank, i)):
#             try:
#                 opened_ptr = checkCudaErrors(cudart.cudaIpcOpenMemHandle(
#                     handles[i], cudart.cudaIpcMemLazyEnablePeerAccess
#                 ))
#                 remote_buffer_ptr_list.append(opened_ptr)
#             except Exception as e:
#                 print(f"Rank {dist_param.local_rank}: Failed to open IPC handle from rank {i}: {e}")
#                 raise
#         else:
#             raise RuntimeError(
#                 f"Rank {dist_param.local_rank} cannot access rank {i}'s memory. "
#                 "Ensure that all ranks are on the same node and can access each other's memory."
#             )
    
#     dist.barrier()
#     return torch_buffer, remote_buffer_ptr_list
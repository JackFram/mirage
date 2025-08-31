import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator

import cutlass
import cutlass.cute as cute

import mpk_cute_dsl.kernel.dsl_ptx_wrapper as inline_ptx

idx_2_name_dict = {
            1: "Hist+All2All",
            2: "Dispatch-Send",
            3: "Dispatch-Recv",
            4: "Fused-FFN-W13",
            5: "Fused-FFN-W2-Send",
            6: "Combine-Recv",
            7: "Fetch-Task",
            8: "Sync-Task",
            9: "Add-Task",
            10: "Terminate-Task",
            11: "Undefined-Task",
            12: "Token-Gather-Task"
        }

name_2_idx_dict = {
            "Hist+All2All": 1,
            "Dispatch-Send": 2,
            "Dispatch-Recv": 3,
            "Fused-FFN-W13": 4,
            "Fused-FFN-W2-Send": 5,
            "Combine-Recv": 6,
            "Fetch-Task": 7,
            "Sync-Task": 8,
            "Add-Task": 9,
            "Terminate-Task": 10,
            "Undefined-Task": 11,
            "Token-Gather-Task": 12
        }

class EventType(Enum):
    kBegin: cutlass.Int32 = 0
    kEnd: cutlass.Int32 = 1
    kInstant: cutlass.Int32 = 2

event_type_map = {
    "begin": EventType.kBegin,
    "end": EventType.kEnd,
    "instant": EventType.kInstant
}


class DslProfiler:
    def __init__(self, profiler_buffer=cute.Tensor, profiler_ptr=cute.Tensor, buffer_size: cutlass.Int32 = 1024, profiler_enabled: bool = False):
        # |block_idx:8b|warp_idx:4b|event_type:2b|event_idx:18b|timestamp:32b|
        self.profiler_buffer = profiler_buffer
        self.profiler_ptr = profiler_ptr
        self.buffer_size = buffer_size
        self.profiler_enabled = profiler_enabled
        
    def __extract_mlir_values__(self):
        return [
            self.profiler_buffer.__extract_mlir_values__(), 
            self.profiler_ptr.__extract_mlir_values__(),
            ]

    def __new_from_mlir_values__(self, values):
        return DslProfiler(values[0], values[1], self.buffer_size, self.profiler_enabled)
    
    @cute.jit
    def profile_event(self, event_name: str, event_type: str):
        if cutlass.const_expr(not self.profiler_enabled):
            return
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if thread_idx % 32 == 0:
            event_id = name_2_idx_dict[event_name] & 0x3FFFF
            event_tag = ((block_idx & 0xFF) << 24 | (warp_idx & 0xF) << 20 | event_type_map[event_type].value << 18 | event_id)
            timestamp_tag = inline_ptx.get_globaltimer_lo()
            full_tag = (event_tag.to(cutlass.Uint64) << 32) | timestamp_tag.to(cutlass.Uint64)
            # Caveats(Zhihao): around 370 ns overhead here for the profiler.
            insert_idx = inline_ptx.atomic_add(self.profiler_ptr, cutlass.Int32(1))
            inline_ptx.st_flag_volatile_64(
                self.profiler_buffer[insert_idx, None],
                full_tag
            )

def decode_tag(tag):
    event_tag = int(tag >> 32)
    timestamp = int(tag & 0xFFFFFFFF)
    
    block_idx = (event_tag >> 24)
    warp_idx = (event_tag >> 20) & 0xF
    event_type = (event_tag >> 18) & 0x3
    event_idx = event_tag & 0x3FFFF

    return (
        block_idx,
        warp_idx,
        event_type,
        event_idx,
        timestamp
    )


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    file_name: str,
    num_sm: int=148,
    num_warps: int=9,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_sm):
        pid = tgen.create_group(f"block_{block_idx}")
        for warp_idx in range(num_warps):
            tid = pid.create_group(f"warp_{warp_idx}")
            tid_map[(block_idx, warp_idx)] = tid

    for i in range(0, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue

        full_tag = profiler_buffer_host[i : i + 1].view(dtype=torch.uint64)
        full_tag = int(full_tag.item())
        block_idx, warp_idx, event_type, event_idx, timestamp = decode_tag(
            full_tag
        )

        event = idx_2_name_dict[event_idx]
        tid = tid_map[(block_idx, warp_idx)]

        if (block_idx, warp_idx, event_idx) in track_map:
            track = track_map[(block_idx, warp_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, warp_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()

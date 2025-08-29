import numpy as np
import torch
import math
import time

def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """
    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return

def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: bool = True,
    l2_flush_size_mb: int = 256,
    l2_flush_device: str = "cuda",
    sleep_after_run: bool = False,
):
    """
    Benchmark kernel execution time without using CUDA graphs.
    Measures kernel launch latency + actual kernel execution time for fn().
    Can flush L2 cache and sleep after the run.

    Number of dry run and actual run iterations can be set by iteration count or time:
    - If dry_run_iters and repeat_iters are provided, provided iteration count will be used.
    - If dry_run_iters and repeat_iters are not provided, dry_run_time_ms and repeat_time_ms will be used.

    Returns an array of measured times so that the caller can compute statistics.

    Args:
        fn: Function to benchmark.
        dry_run_iters: Number of dry runs during which times does not count. If not provided, dry_run_time_ms will be used.
        repeat_iters: Number of iterations. If not provided, repeat_time_ms will be used.
        dry_run_time_ms: Time to run the dry run in milliseconds.
        repeat_time_ms: Time to run the repeat in milliseconds.
        l2_flush: Whether to flush L2 cache.
        l2_flush_size_mb: Size of the L2 cache to flush.
        l2_flush_device: Device that needs to flush L2 cache.
        sleep_after_run: Whether to sleep after the run. Sleep time is dynamically set.

    Returns:
        measured_times: List of measured times.
    """

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    if l2_flush:
        l2_flush_size = int(l2_flush_size_mb) * 1024 * 1024
        buffer = torch.empty(l2_flush_size, device=l2_flush_device, dtype=torch.int8)

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.cuda.synchronize()
    fn()  # Call once to exclude initial overhead
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        if l2_flush:
            buffer.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    print("estimated kernel execution time: {:.3f} ms".format(estimated_kernel_execution_time))

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.cuda.synchronize()
    for _ in range(dry_run_iters):
        if l2_flush:
            buffer.zero_()
        fn()
    torch.cuda.synchronize()

    # Actual run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.cuda.synchronize()
    for iter_idx in range(repeat_iters):
        if l2_flush:
            buffer.zero_()
        start_events[iter_idx].record()
        fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.cuda.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    return measured_times


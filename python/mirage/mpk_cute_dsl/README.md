# Mirage Persisitent Kernel (MPK) Blackwell IntraNode MoE

A SM100 IntraNode expert parallelism MoE kernel implementation with CuteDSL.

### Installing

```bash
git clone --branch blackwell-moe https://github.com/JackFram/mirage.git
cd mirage/python/mirage
pip install -e .
```

### Running the demo
It requires at least two NVIDIA SM100 GPUs to run the demo.
```bash
cd mpk_cute_dsl/
python mpk_intranode_moe.py
```

## Design

Expert parallelism requires three operations for the MoE: **Dispatch**, **Exepert FFN**, and **Combine**. Conventional implementation 
would have implemented three standalone kernels for each, while in this demo, we are trying to fuse them into a single kernel (also known as Mega-Kernel).
This then benefits from reduced kernel launch/cleanup overhead and more importantly, allow fine-grained task synchronizations that can utilize GPU resources
more efficiently. 

To learn more about the Mega-Kernel design, please refer to [MPK blog](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17).

### Scheduler
[**mpk_scheduler**](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/mpk_scheduler.py)

The kernel runtime scheduler for handling task fetch and synchronization. For each SM, we have a specialized scheduler warp while having eight warps as worker warps.
We maintain a queue structure in global memory for adding and fetching tasks with a consumer and producer pointer.

**Task fetch**:
```python
# get the next task position in the queue
task_load_idx = inline_ptx.atomic_add(
                    task_consume_idx,
                    cutlass.Int32(1),
                ) % cutlass.Int32(mpk_queue_len)
# waiting for the task to be produced and fetch
while(cutlass.dynamic_expr(task_code == 0)): 
    # Wait task update if task_code == 0 (fetch)
    # TODO(Zhihao): try ld.relax and also measure the overhead (might slow down works on other warps)
    task_desc = inline_ptx.ld_flag_relaxed_gpu_u32(task_queue[task_load_idx, None])
    task_code = task_desc >> 28
```
**Task synchronization** is done through shared memory message passing.

### Tasks

We have 7 main tasks:

  - [HistAll2All](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/hist_a2a_task.py)
  - [Dispatch-Send](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/dispatch_send_task.py)
  - [Dispatch-Recv](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/dispatch_recv_task.py)
  - [Token-Gather](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/token_gather_task.py)
  - [Fused-FFN-W13](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/fused_ffn_w13_task.py)
  - [Fused-FFN-W2-Send](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/fused_ffn_w2_send_task.py)
  - [Combine-Recv](https://github.com/JackFram/mirage/blob/blackwell-moe/python/mirage/mpk_cute_dsl/kernel/mpk_task_kernel/combine_recv_task.py)

**HistAll2All**: A single task that check the send count of each **dispatch-send** worker and notify the remote rank.
  - collect the token dispatch info based on the topk_indices tensor
  - notify each remote expert on the completion of the send

**Dispatch-Send**: Send each token to its corresponding expert buffer (either remote or local).
  - send each individual token to the remote expert buffer without synchronization 
  - only mark the completion of the current token with the **HistAll2All** task through atomic operation over local device memory

**Dispatch-Recv**: Collect the recieved token counts for each expert and dynamically add **token-gather** task into task queue. 
  - each task will receive all the tokens for a local expert from a remote rank, so there is num_local_experts * num_ranks tasks in total
  - collect token infos at the receiver end and also add the corresponding **token-gather** tasks.

**Token-Gather**: Gather tokens from different ranks for the local experts and dynamically add **fused-ffn-w13** tasks to the queue based on the gathered counts.
  - each task will transfer the token from the communication buffer to a continuous chunk of global memory that is the input for the **fused-ffn-w13** task
  - each task will also dynamically add the **fused-ffn-w13** task once certain number of tokens have already arrived.

**Fused-FFN-W13**: Fused W13 GeMM + SwapAB + SwiGLU and add **fused-ffn-w2-send**, **combine-recv**, and **terminate** task to the queue. 
  
  - We fuse the W1 and W3 GeMM into a single GeMM with a novel layout (W13-interleaved) where we can directly do the SwiGLU op in the register memory during the epilogue stage.
  - Dynamically add **fused-ffn-w2-send** to the task queue based on the completion of the current token tiles.
  - Add **combine-recv**, and **terminate** task to the queue after all the **fused-ffn-w2-send** tasks have been added.

**Fused-FFN-W2**: Fused W2 GeMM + SwapAB + send back the tokens to its corresponding rank. 
  - W2 GeMM with SwapAB
  - Send the tiled output directly to the communication buffer 

**Combine-Recv**: Recieve tokens from the output of FFN tasks with weighted summation.
  - Each token is waiting for all its topk output from FFN tasks to return
  - Then reduction is performed where the results are stored to the output tensor. 

<!-- ROADMAP -->
## Roadmap

- [ ] Correctness and performance testing and benchmarking
    - [ ] Add acq/rel pattern for data consistency
    - [ ] Correct shared memory layout for swapAB case (W13 task)
    - [ ] Optimize dispatch send/recv latency
- [ ] Support for Ampere and Hopper
- [ ] Low precision support (e.g., fp8)
- [ ] Add MPK counter reset in the termination task
- [ ] Support prefilling MoE 

<!-- ## Acknowledgments

  -  -->

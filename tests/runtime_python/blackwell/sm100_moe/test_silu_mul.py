import torch
import runtime_kernel_blackwell

torch.set_printoptions(sci_mode=False, profile="full")
# torch.set_printoptions(sci_mode=False)

g = torch.Generator(device="cuda").manual_seed(1234)

iter_dim = 4096
top_ks = [64]
batch_sizes = [32]

has_residual = True

for top_k in top_ks:
    for batch_size in batch_sizes:
        print(
            f"\n=== Testing batch_size = {batch_size} ==="
        )

        w1x_w3x = torch.randn((batch_size, top_k, iter_dim), device="cuda", dtype=torch.bfloat16)
        output = torch.empty(batch_size, top_k, iter_dim, device="cuda", dtype=torch.bfloat16)


        runtime_kernel_blackwell.linear_sm100_mpk(w1x_w3x, output) # with residual and swapAB


        with torch.no_grad():
            w1x = w1x_w3x[:,:,:iter_dim]
            w3x = w1x_w3x[:,:,iter_dim:]

            # SwiLU multiplication
            torch_out = torch.nn.functional.silu(w1x) * w3x

        torch.testing.assert_close(
            output,
            torch_out,
            rtol=1e-2,
            atol=1e-2,
        )
        print("Test passed!")

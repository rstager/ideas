# train.cpp: High-Performance Training Loop

## Motivation

The `autoresearch` project has a single-file GPT pretraining script (`train.py`, ~1164 lines) built on PyTorch. It is single-GPU and already fairly optimized (flash attention, bf16, fused RMS norm), but Python-level overhead and framework indirection still leave performance on the table. The goal is to use Claude to translate `train.py` into a faster implementation in C++ (primary target), with JAX or Rust as secondary options worth evaluating.

## What train.py does

- Defines a GPT model (~30M–100M param range) with:
  - Grouped-query attention with flash attention kernels
  - Rotary positional embeddings
  - Value residual connections (ResFormer-style)
  - Per-layer residual and x0 scaling lambdas
  - ReLU-squared MLP activation
  - Heterogeneous block configs (variable width, sliding window)
- Single-GPU training loop with gradient accumulation
- Muon + Adam hybrid optimizer
- Cosine-then-linear LR schedule with warmup
- WandB logging, eval via BPB on validation set
- Data loading via `prepare.py` (tokenized shards)

## Target: C++ with CUDA

The primary target is a C++ implementation using CUDA, following the approach of [llm.c](https://github.com/karpathy/llm.c). Key components:

- **CUDA kernels**: Flash attention (use cutlass or load FA2/FA3 as a library), fused RMS norm, rotary embedding, ReLU-squared MLP, embedding lookup
- **Memory management**: Manual bf16 buffer allocation, reuse activations across microsteps
- **Optimizer**: Muon (SVD-based) is the tricky part; needs cuSOLVER or a custom SVD kernel. Adam is straightforward.
- **Data pipeline**: Memory-mapped tokenized shards, no Python overhead

### Expected speedups

- Eliminate Python interpreter + PyTorch dispatch overhead (~5-15% on small models)
- Fuse more operations (norm + linear, attention + rotary)
- Tighter memory layout and fewer allocations
- Potentially significant for the many small ops (scaling lambdas, value residual gating)

## Alternative: JAX/XLA

JAX with `jit` would capture the full computation graph and fuse aggressively without manual kernel work. Advantages:

- Much less effort than C++ (still Python, but compiled)
- XLA fusion handles small-op overhead automatically
- Easy to extend to multi-GPU via `pjit`/`shard_map`

Disadvantage: harder to beat hand-tuned CUDA kernels for attention, and JAX ecosystem is less familiar.

## Alternative: Rust with CUDA bindings

Rust via `cudarc` or `candle` gives memory safety without Python overhead. Middle ground between C++ and JAX in terms of effort and control. Worth considering if the project grows beyond a single training script.

## Approach

1. **Benchmark `train.py`** to establish baseline tokens/sec and identify where time is actually spent (is it attention? data loading? optimizer? small ops?)
2. **Pick the target** based on profiling results -- if most time is in flash attention and matmuls, C++ gains are marginal and JAX is the better ROI. If Python/dispatch overhead is significant, C++ wins.
3. **Use Claude to translate** the model and training loop, starting with the forward pass and verifying numerical equivalence against the Python version on a fixed seed.
4. **Incrementally port** optimizer, data loading, logging, and eval.
5. **Validate** by training to the same BPB on the same data and comparing loss curves.

## Open questions

- Is Muon's SVD step a bottleneck that needs special attention in C++?
- Should the heterogeneous block configs (variable n_embd, sliding window) be templated at compile time or handled dynamically?
- Is it worth keeping a Python wrapper (pybind11) for experiment management and WandB, with only the hot loop in C++?

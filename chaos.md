# Chaos: Dynamic Block Scheduling for Transformers

## Core idea

Instead of a fixed sequence of transformer blocks, every attention block and every MLP is a free-floating unit that can act on the residual stream at any time, in any order, any number of times. A learned router dynamically decides:

1. **Which block** runs next on the residual stream
2. **When to stop** processing and emit the output

Blocks are no longer tied to a position in a static stack. The computation graph is determined at runtime per-token (or per-sequence).

## Depth embedding

Since blocks no longer have a fixed layer index, we inject a **depth embedding** into the residual stream -- a learned or positional encoding of how many compute steps have been applied so far. This gives the router and each block information about where the token is in its processing lifecycle.

- Could be a simple learned embedding table indexed by step count
- Or a continuous signal (sinusoidal, like positional encoding but over depth instead of sequence position)
- Added to (or concatenated with) the residual stream before each block application

## Router design

The router takes the current residual state (plus depth embedding) and produces:

- A **block selection** distribution over all available attention and MLP blocks
- A **halt probability** indicating whether to stop processing this token

This is similar to Adaptive Computation Time (ACT) and mixture-of-experts routing, but applied to the temporal scheduling of blocks rather than just gating experts within a layer.

### Routing strategies

- **Hard routing**: Pick one block per step (like top-1 MoE). Requires straight-through estimators or REINFORCE for gradients.
- **Soft routing**: Weighted combination of block outputs. Differentiable but expensive (runs all blocks, defeats the efficiency purpose).
- **Top-k routing**: Run k blocks per step. Balances expressiveness and cost.
- **Halting**: Accumulate halt probability across steps (ACT-style). When cumulative halt exceeds a threshold, stop. Adds a ponder cost penalty to the loss.

## Why this might work

- **Adaptive compute**: Easy inputs exit early, hard inputs get more processing. The model learns its own depth.
- **Block reuse**: A single well-trained attention head could be applied multiple times at different depths, like an iterative refinement. Fewer parameters for equivalent effective depth.
- **Heterogeneous routing**: Some tokens might need more attention, others more MLP. The router can specialize the compute path per token.
- **Emergent structure**: The model might learn something resembling a fixed pipeline for common cases but deviate for unusual inputs.

## Concerns and open questions

- **Training stability**: Discrete routing decisions are hard to train. ACT-style halting helps but adds complexity. The router might collapse to a fixed ordering, recovering a standard transformer with extra overhead.
- **Efficiency**: If the router is cheap but blocks are expensive, dynamic dispatch could still be fast. But variable-length computation per token complicates batching -- tokens in the same batch may halt at different steps, requiring padding or bucketing.
- **Load balancing**: Need auxiliary losses to prevent the router from starving some blocks and overusing others (same problem as in MoE).
- **Depth embedding fidelity**: Does step count carry enough information? Might also need to encode which blocks have already been applied (a "computation history" embedding).
- **Comparison to existing work**: Universal Transformers share weights across depth. MoE routes between experts within a layer. ACT adapts depth. This combines all three ideas. Worth surveying what has been tried in this intersection.

## Relation to autoresearch

The `train.py` architecture already has heterogeneous block configs (variable width, sliding windows, optional value residuals). Chaos would generalize this further -- instead of a hand-designed block schedule, let the model learn its own.

## Minimal experiment

1. Take the existing autoresearch GPT with N blocks
2. Remove the fixed ordering -- pool all attn + MLP modules
3. Add a small router MLP (residual + depth embedding -> block logits + halt logit)
4. Train with ACT-style halting loss (ponder cost)
5. Compare loss-per-FLOP against the standard fixed-order model
6. Visualize learned routing patterns -- does structure emerge?

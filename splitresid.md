# Split Residual: Multiple Independent Residual Streams

## Core idea

Standard transformers superimpose all information onto a single residual stream vector. Every block reads from and writes to the same shared space. As the residual dimension grows, this creates noise -- unrelated features interfere with each other, and blocks must learn to ignore the majority of the stream that is irrelevant to their function.

Split Residual partitions the residual stream into multiple independent sub-streams. Each block does not see the full residual -- instead, a learned routing function selects which sub-streams a block reads from and writes to. This reduces the effective dimensionality each block operates on and eliminates cross-talk between unrelated features.

## Why this matters for large residuals

In wide models (n_embd = 4096+), the residual stream carries hundreds of superimposed concepts. Each attention head or MLP must:

1. Project out the tiny subspace it cares about (wasting parameters on near-zero projections)
2. Write its output back into the full space (risking interference with other features)

With split streams, a block only touches the sub-streams routed to it. This means:

- **Less noise**: A block operating on a 256-dim sub-stream has a much cleaner signal than one projecting from 4096 dims
- **Fewer parameters**: Input/output projections scale with sub-stream width, not full model width
- **Better gradient flow**: Gradients propagate through fewer superimposed pathways

## Architecture

### Stream layout

Split the residual of width D into K sub-streams of width D/K. For example, D=2048 with K=8 gives eight 256-dim streams.

### Routing functions

Before each block:
- **Input router**: Selects and concatenates a subset of sub-streams to form the block's input
- **Output router**: Distributes the block's output back to the appropriate sub-streams (additive, like a standard residual connection)

Routing can be:
- **Fixed**: Predetermined patterns (e.g., block 0 reads streams 0-3, block 1 reads streams 2-5). Simple, no overhead, but inflexible.
- **Learned static**: Routing weights trained but fixed at inference. A soft attention over sub-streams, baked in after training.
- **Dynamic**: Input-dependent routing (like MoE). Most expressive but adds compute.

### Cross-stream communication

Fully independent streams can't share information. Options:
- **Periodic mixing layers**: Every N blocks, apply a cheap cross-stream linear or attention
- **Overlapping routes**: Blocks read from overlapping subsets, providing implicit mixing
- **Dedicated bridge blocks**: Specific blocks whose job is cross-stream transfer

## Phase 1: Theoretical analysis

Before implementing, establish whether the hypothesis holds:

1. **Measure superposition noise in existing models**: Take a trained GPT, record residual stream activations, and analyze:
   - How many principal components carry meaningful signal vs. noise at each layer?
   - Do different attention heads read from orthogonal subspaces? (If they already do, split residual formalizes what the model learns implicitly.)
   - What is the effective rank of the residual stream at each layer?

2. **Interference quantification**: For each block, measure how much of its input projection energy is "wasted" on dimensions irrelevant to its output. This is the noise that split residual would eliminate.

3. **Information flow analysis**: Track which dimensions are written by early layers and read by later layers. If the flow is sparse (most layers only interact with a subset), split residual is a natural fit.

## Phase 2: Visualization

Build tools to visualize:

- **Stream utilization heatmaps**: Which sub-streams are active at each layer (if routing were applied post-hoc to a standard model)
- **Cross-stream dependency graphs**: Which sub-streams need to communicate, and how often
- **Effective rank per layer**: How the information capacity of the residual evolves through depth
- **Comparison plots**: Standard residual vs. split residual signal-to-noise ratio

These visualizations should produce a clear picture of whether splitting is justified before writing any training code.

## Phase 3: Implementation and testing with nanoGPT

Use nanoGPT as the base for experiments:

1. **Baseline**: Train standard nanoGPT at a chosen scale (e.g., 124M params, OpenWebText)
2. **Split residual variant**: Same parameter budget, but with K sub-streams and routing
3. **Ablations**:
   - Vary K (2, 4, 8, 16) -- what is the right granularity?
   - Fixed vs. learned vs. dynamic routing
   - Cross-stream mixing frequency
   - Sub-stream width uniformity vs. heterogeneous widths
4. **Metrics**: Loss-per-FLOP, loss-per-parameter, throughput (tokens/sec), convergence speed

## Relation to existing work

- **Mixture of Experts**: MoE routes tokens to experts; split residual routes features to blocks. Orthogonal ideas that could combine.
- **Multi-head latent attention**: Compresses KV into a latent space, somewhat analogous to operating on a subspace.
- **Sparse attention / block-sparse**: Reduces attention compute but doesn't address residual superposition.
- **Product key memory / multi-stream architectures**: Some prior work on parallel residual streams exists (e.g., parallel attention + MLP in PaLM) but typically not with learned routing.

## Open questions

- Does splitting hurt tasks that require global integration of many features (e.g., long-range reasoning)?
- What is the compute overhead of routing, and does it pay for itself in reduced block size?
- Can dynamic routing learn meaningful specialization, or does it collapse to a fixed pattern?
- Is there a connection to the "features as directions" interpretation from mechanistic interpretability? Split streams might correspond to feature groups.

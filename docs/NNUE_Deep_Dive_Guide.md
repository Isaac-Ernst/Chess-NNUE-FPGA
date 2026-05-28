# NNUE Deep Dive — Everything You Need to Know

**For:** Isaac Ernst — Chess NNUE on FPGA Project
**Prerequisite:** Basic ML class (loss functions, gradient descent, neural networks)
**Date:** May 28, 2026

---

## Table of Contents

1. [What an NNUE Actually Is (and Isn't)](#1-what-an-nnue-actually-is-and-isnt)
2. [How NNUE Fits Into a Search Algorithm](#2-how-nnue-fits-into-a-search-algorithm)
3. [The Loss Function — What Are We Optimizing?](#3-the-loss-function--what-are-we-optimizing)
4. [W/L/D vs Stockfish Evals — Training Target Analysis](#4-wld-vs-stockfish-evals--training-target-analysis)
5. [Dataset Sizing — How Much Data Do You Need?](#5-dataset-sizing--how-much-data-do-you-need)
6. [Learning Rate — Why It Matters and How to Choose It](#6-learning-rate--why-it-matters-and-how-to-choose-it)
7. [Training Cycles and Curriculum — Order Matters](#7-training-cycles-and-curriculum--order-matters)
8. [Power-of-2 Layer Sizes and FPGA Memory Allocation](#8-power-of-2-layer-sizes-and-fpga-memory-allocation)
9. [Memory Hierarchy — BRAM, SRAM, DDR4, VRAM, SSD, MicroSD](#9-memory-hierarchy--bram-sram-ddr4-vram-ssd-microsd)
10. [Design Choices Explained](#10-design-choices-explained)
11. [Your Testing Methodology — The Right Approach](#11-your-testing-methodology--the-right-approach)
12. [Implementation Steps — From Zero to Working NNUE](#12-implementation-steps--from-zero-to-working-nnue)
13. [Common Pipeline Pitfalls (Read Before Building)](#13-common-pipeline-pitfalls-read-before-building)

---

## 1. What an NNUE Actually Is (and Isn't)

### The Name
NNUE = **Efficiently Updatable Neural Network** (yes, the acronym is reversed — it's from Japanese, 「NNUE評価関数」). Invented by Yu Nasu for Shogi in 2018, adopted by Stockfish in 2020.

### What It Is
A **very small, very fast neural network** (2-4 layers, ~200K to ~40M parameters) that takes a chess position as input and outputs a single number: **how good is this position for the side to move?** That number is called the **evaluation score**, measured in **centipawns** (1 pawn = 100 centipawns).

```
Position → [Feature Encoding] → [Neural Network] → Score (e.g., +145 centipawns)
                                                     "White is up about 1.5 pawns"
```

### What It Is NOT
- **Not a move selector.** It doesn't pick moves — the search algorithm does that.
- **Not a deep network.** GPT-4 has ~1.8 trillion parameters. Your BRAM NNUE has ~200K. It's closer to logistic regression than to a transformer.
- **Not doing attention/transformers.** Despite you mentioning "attentions" — NNUEs use simple fully connected (linear) layers with ClippedReLU. No attention mechanism, no self-attention, no transformer blocks. This is good news: the math is just matrix multiplication and clamping.
- **Not an end-to-end system.** It's one component (the evaluation function) inside a larger system (the search engine).

### The Key Insight: Incremental Updates
In a chess game, each move changes only 2-4 pieces on the board. A naive approach would recompute the entire first layer (768 or 40,960 multiplications) after every move. The NNUE trick:

```
Before move e2-e4:
  accumulator = W[pawn_e2] + W[pawn_d2] + W[knight_g1] + ... (all 16-32 pieces)

After move e2-e4:
  accumulator = accumulator - W[pawn_e2] + W[pawn_e4]    ← only 2 operations!
```

Instead of recomputing from scratch, you **subtract the row for the removed feature and add the row for the new feature**. This is why the first layer is called the "accumulator" — it accumulates changes incrementally.

This is the entire reason NNUEs exist. Without this trick, you'd just use a regular neural network. With it, the first layer (which contains 95%+ of the parameters) costs almost nothing per move.

### The Architecture You're Building

```
BRAM-only (small) — SHARED accumulator across perspectives:
  Input:  768 binary features per perspective (12 piece types × 64 squares)
          Sparse: ~16-32 active features per perspective
  Layer 0: 768 → 256   shared linear layer, applied to BOTH perspectives independently
                       Per-perspective output: 256 × int16 (accumulator state)
                       After ClippedReLU + concat (STM-first, see §13.1): 512 uint8
  Layer 1: 512 → 32    fully connected
  Layer 2: 32 → 32     fully connected
  Layer 3: 32 → 1      signed centipawn output
  Activation: ClippedReLU between layers (no activation on final output)

  Total parameters: ~213K (shared accumulator counted once, plus biases)
  Weight storage:   ~210 KB int8 (fits comfortably in FPGA BRAM)

DDR4 (large) — SHARED accumulator across perspectives:
  Input:  40,960 features per perspective (HalfKP: king position × piece-square)
          Sparse: ~30 active features per perspective
  Layer 0: 40,960 → 1024  shared linear layer
                          Per-perspective output: 1024 × int16
                          After ClippedReLU + concat (STM-first): 2048 uint8
  Layer 1: 2048 → 8
  Layer 2: 8 → 32
  Layer 3: 32 → 1

  Total parameters: ~42M (shared accumulator counted once)
  Weight storage:   ~40 MB int8 (lives in DDR4, BRAM holds intermediate state)
```

The accumulator is "shared" in the sense that the same `768 × 256` weight matrix is applied to both white's perspective and black's perspective. This halves parameter count vs a non-shared design. The trick that makes this work is **concatenating in side-to-move order** (see §13.1), so the network can always tell whose turn it is.

---

## 2. How NNUE Fits Into a Search Algorithm

### The Big Picture

A chess engine has two parts that work together:

```
┌─────────────────────────────────────────────────────┐
│  SEARCH ALGORITHM (alpha-beta)                       │
│                                                      │
│  "What happens if I play e4, then they play e5,      │
│   then I play Nf3, then they play Nc6..."            │
│                                                      │
│  Explores a tree of moves, depth-first.              │
│  At each LEAF NODE, asks the evaluator:              │
│  "How good is this position?"                        │
│                                                      │
│         ┌─────────────┐                              │
│         │ NNUE EVAL   │  ← called millions of times  │
│         │ position →  │     per second                │
│         │   → score   │                              │
│         └─────────────┘                              │
│                                                      │
│  Uses the scores to decide which move is best.       │
└─────────────────────────────────────────────────────┘
```

### Where Exactly Is NNUE Called?

In your alpha-beta search, at every **leaf node** (the deepest point of each branch):

```python
def alpha_beta(position, depth, alpha, beta):
    if depth == 0:
        return nnue_evaluate(position)   # ← HERE. This is the ONLY place NNUE runs.
    
    for move in generate_moves(position):
        position.make_move(move)
        score = -alpha_beta(position, depth - 1, -beta, -alpha)
        position.unmake_move(move)
        
        if score >= beta:
            return beta          # beta cutoff (pruning)
        if score > alpha:
            alpha = score
    
    return alpha
```

### The Accumulator Stack

Since search explores moves and then undoes them (make_move / unmake_move), the accumulator must be saved and restored efficiently:

```
Search depth 0: accumulator_0 = [initial position]
  make_move(e2e4)
Search depth 1: accumulator_1 = accumulator_0 - W[pawn_e2] + W[pawn_e4]
  make_move(e7e5)
Search depth 2: accumulator_2 = accumulator_1 - W[pawn_e7] + W[pawn_e5]
  ← NNUE evaluates here using accumulator_2
  unmake_move(e7e5)
  ← back to accumulator_1 (just pop the stack)
  make_move(d7d5)
Search depth 2: accumulator_2 = accumulator_1 - W[pawn_d7] + W[pawn_d5]
  ← NNUE evaluates here
  unmake_move(d7d5)
  ...
```

You maintain a **stack of accumulators**, one per search depth. make_move pushes a copy and updates it. unmake_move pops it. The cost is a copy of 256 int16 values (512 bytes) per ply — trivial.

### Why Speed Matters

At depth 20 with good pruning, a chess engine examines ~1-50 million positions per second. Each position needs one NNUE evaluation. If your NNUE takes 1 microsecond, that's 1M evals/sec — acceptable. If it takes 1 millisecond (like reading from an SSD), that's 1,000 evals/sec — your engine is dead.

This is why NNUE lives on the FPGA (nanoseconds per eval) and not on the SSD.

---

## 3. The Loss Function — What Are We Optimizing?

### The Problem Statement

You have:
- **Input:** A chess position (represented as features)
- **Target:** Some measure of "how good this position is"
- **Goal:** Train a neural network to predict the target from the input

The loss function measures how wrong your predictions are. You minimize it with gradient descent.

### Option 1: MSE on Centipawn Scores (Regression)

```
Loss = mean( (predicted_score - target_score)² )
```

Where `target_score` is the centipawn evaluation from some source (Stockfish, or your own engine's search).

**Why it works:** You're directly learning to output centipawn values. The network learns "this position is worth +150 centipawns" — a fine-grained, continuous signal.

**Problem:** Where do target scores come from? If from Stockfish, you're training to mimic Stockfish (see Section 4). If from your own engine, you have a bootstrapping problem (your engine needs an eval to generate evals).

### Option 2: Cross-Entropy on Game Outcome (Classification)

```
Target: game result from side-to-move's perspective
  Win  = 1.0
  Draw = 0.5
  Loss = 0.0

predicted = sigmoid(raw_network_output)  # squash to [0, 1]
Loss = -mean( target * log(predicted) + (1 - target) * log(1 - predicted) )
```

This is binary cross-entropy where 0.5 means "equal position" and 1.0 means "winning."

**Why it works:** Game outcomes are **ground truth** — they're the actual result of the game. No circular dependencies. The network learns "positions that look like this tend to lead to wins/draws/losses." The sigmoid output can be converted to centipawns later:

```
centipawns = 400 * log10(p / (1 - p))    # where p is the sigmoid output
```

**Problem:** Coarse signal — a position where one side is up a rook and a position where they're up a pawn might both result in a win. The network has to figure out the magnitude from the statistical pattern.

### Option 3: Combined Loss (What Stockfish Uses)

```
Loss = λ * MSE(predicted, search_score) + (1-λ) * CrossEntropy(predicted, game_result)
```

Where λ is a blending parameter (typically 0.5-0.75 favoring the search score).

**Why it works:** You get the fine-grained signal from search scores AND the ground-truth anchoring from game outcomes. This is the strongest approach once you have a working search engine generating scores.

### What I Recommend for You

| Phase | Loss Function | Why |
|-------|--------------|-----|
| First training (BRAM net) | Cross-entropy on game outcomes (W/L/D) | Simple, no dependencies, establishes baseline |
| Refinement | MSE on your own engine's search scores | Self-improvement loop once you have a working search |
| Final polish | Combined loss (70% search + 30% game outcome) | Best of both worlds |

---

## 4. W/L/D vs Stockfish Evals — Training Target Analysis

This is the most important question you asked, and your intuition about training on Stockfish evals is actually quite sharp. Let me be fully honest.

### The Case FOR Training on Stockfish Evaluations

**Your argument:** "If I train my small network to predict Stockfish's centipawn scores, won't it learn to be as close to Stockfish as possible, giving me the strongest small eval?"

**Yes, this is essentially knowledge distillation, and it works.** This is literally what `nnue-pytorch` (Stockfish's training code) does — they use Stockfish at depth 8-12 to evaluate billions of positions, then train the NNUE to predict those scores. The small network learns a compressed version of Stockfish's evaluation.

**Advantages of Stockfish eval targets:**
- Fine-grained signal (centipawn precision, not just W/L/D)
- Each position gets a score proportional to its advantage (not just binary)
- Faster convergence (stronger gradient signal per position)
- Your small net would likely be 50-100 ELO stronger than W/L/D-only training

### The Case AGAINST (for YOUR project specifically)

**1. Philosophical: "Everything from scratch"**
Your admissions pitch is "I didn't use Stockfish or any existing engine." If your NNUE is literally trained to mimic Stockfish's evaluation, that story weakens. An admissions reviewer who understands ML will ask: "So your neural network is a compressed copy of Stockfish?"

**2. Practical: You need Stockfish to generate the training data**
To get Stockfish evaluations, you need to run Stockfish at depth 8-12 on every training position. For 200M positions, that takes:
- ~0.1 seconds per position at depth 10
- 200M × 0.1s = 20M seconds = **231 days** on a single core
- Even with 8 threads on your PC: ~29 days of continuous Stockfish running
- This is a significant compute cost before you even start training

**3. You cap your ceiling at Stockfish's evaluation**
Your network can never be *better* than Stockfish's evaluation on those positions — it can only be a lossy approximation. With W/L/D training, your network has the theoretical ability to discover evaluation patterns that Stockfish misses (though in practice this is rare for a small net).

**4. Stockfish eval bias**
Stockfish's evaluation at depth 10 is imperfect. It has biases — it overvalues certain piece configurations, undervalues others. Your network would inherit those biases but wouldn't have Stockfish's deep search to compensate.

### The Honest Answer

**If your ONLY goal were maximum ELO with a small net, you should train on Stockfish evals.** Full stop.

**But your goal is a complete, from-scratch project that impresses admissions boards.** So here's the balanced approach:

```
Phase 1: Train on W/L/D game outcomes (100% independent, your own data pipeline)
         → This is YOUR neural network, trained YOUR way

Phase 2: Self-play refinement (train on your own engine's search scores)
         → Still 100% yours — the scores come from YOUR search + YOUR net

Phase 3 (optional): Knowledge distillation from the large net → small net
         → The large net is also yours, so this is still your own work

Phase 4 (comparison only): Train one version on Stockfish evals as a BASELINE
         → Label it clearly as "Stockfish-distilled baseline" in your paper
         → Compare: "My independently trained net achieves X% of Stockfish-distilled performance"
         → This is actually a STRONGER paper than just having one version
```

### The Real-World Secret

Most competitive NNUE training uses a **blend**: the target is a mix of the search score (from the engine's own search) and the game outcome. Stockfish uses roughly:

```
target = 0.75 * sigmoid(search_score / 400) + 0.25 * game_result
```

This gives fine-grained signal (from search) anchored to ground truth (from game result). You can implement this once you have your own search engine running.

---

## 5. Dataset Sizing — How Much Data Do You Need?

### The Rule of Thumb

A neural network needs roughly **10× to 100× more training examples than it has parameters** to generalize well. Below that, it overfits (memorizes the training data). Above that, returns diminish.

| Model | Parameters | Minimum Data | Ideal Data | Overkill |
|-------|-----------|-------------|-----------|----------|
| BRAM (768→256→32→32→1) | ~200K | 2M positions | 50-200M | 500M+ |
| DDR4 (40960→1024→8→32→1) | ~40M | 400M positions | 1-2B | 5B+ |

### Why These Numbers?

**Underfitting zone (too little data):**
With 200K parameters and only 100K training positions, the network has more "storage capacity" than data. It will memorize each position exactly but fail on new positions. Validation loss will be much higher than training loss.

**Sweet spot:**
At 50-100× parameters, the network is forced to learn **patterns** rather than memorize positions. It learns things like "a rook on an open file is good" or "doubled pawns are weak" — generalizable knowledge.

**Diminishing returns:**
After ~500× parameters, each additional training position adds almost nothing. The network has already learned all the patterns it can represent with its architecture. To improve further, you need a bigger network, not more data.

### How to Split Your Data

```
Total collected positions: e.g., 250M

Training set:    200M (80%)  — used for gradient descent
Validation set:   25M (10%)  — checked after each epoch to detect overfitting
Test set:         25M (10%)  — used ONCE at the very end to report final performance

CRITICAL: Test set is NEVER used during training or hyperparameter tuning.
          It's your honest measurement of generalization.
```

### How to Know If You Have Enough

Plot **training loss** and **validation loss** over epochs:

```
Loss
  │
  │ \
  │  \  ← training loss keeps going down
  │   \___________
  │    \           ← GOOD: val loss follows training loss closely
  │     \__________  (you have enough data)
  │
  └─────────────────── Epoch

  vs.

Loss
  │
  │ \
  │  \  ← training loss keeps going down
  │   \___________
  │    \
  │     \_________
  │        /       ← BAD: val loss starts going UP
  │       / ← This is overfitting. You need more data or a smaller model.
  └─────────────────── Epoch
```

**If validation loss plateaus but doesn't rise:** You have enough data. Adding more won't help much.
**If validation loss rises while training loss falls:** You're overfitting. Get more data, or reduce model size, or add regularization (dropout, weight decay).
**If both losses are still falling at the end:** Train longer or get more data — you haven't converged yet.

### Practical Advice for Your Project

**BRAM net (200K params):**
- Start with 10M positions to validate your pipeline works
- Scale to 50M to get a decent model
- Final training on 200M for best quality
- Each experiment on A2000: 10M = ~30 min, 50M = ~2-3 hrs, 200M = ~8-15 hrs

**DDR4 net (40M params):**
- Validate pipeline locally with 1M positions (just check it doesn't crash)
- Azure Run 1: 200M positions (validate HalfKP features at scale)
- Azure Run 2-3: 500M positions (real training)
- Azure Final: 1B+ positions (push for maximum quality)

---

## 6. Learning Rate — Why It Matters and How to Choose It

### What Is the Learning Rate?

The learning rate (LR, often written as η or α) controls **how big a step** you take when updating weights:

```
new_weight = old_weight - LR × gradient
```

- **LR too high (e.g., 0.1):** Steps are too big. You overshoot the minimum, loss oscillates wildly or diverges to infinity. Like trying to park a car by flooring the gas.
- **LR too low (e.g., 1e-7):** Steps are too small. Training takes forever. You might get stuck in a bad local minimum. Like parking by moving 1mm at a time.
- **LR just right (e.g., 1e-3):** Converges efficiently to a good minimum. This is where the art is.

### Why 1e-3 (0.001)?

For Adam optimizer (which you should use), 1e-3 is the widely tested default. Adam adapts per-parameter learning rates internally, so it's more robust to the initial LR choice than plain SGD. But 1e-3 is still a starting point, not gospel.

### Learning Rate Schedules — Why Decay?

As training progresses, you want to take smaller steps (fine-tuning near the minimum instead of overshooting it):

**Cosine Annealing (recommended):**
```
LR(epoch) = LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(π * epoch / total_epochs))

Starts at LR_max (1e-3), smoothly decays to LR_min (~1e-6)
Shape looks like the first half of a cosine wave
```

Why cosine? It decays slowly at first (good — still exploring), then quickly in the middle (narrowing in), then slowly again at the end (fine-tuning). Empirically works well across many tasks.

**Warmup (optional, recommended for large models):**
```
Epochs 1-5:    LR ramps from 1e-5 up to 1e-3    (warmup)
Epochs 5-100:  LR decays from 1e-3 down to 1e-6  (cosine decay)
```

Why warmup? At the start of training, gradients can be wild (random weights → random gradients). A low LR prevents destructive early updates. After a few epochs, the weights are in a reasonable region and you can safely use the full LR.

### How to Choose — Practical Steps

1. **Start with Adam, LR=1e-3, cosine decay over your total epochs**
2. Run a short training (5 epochs on 10M positions)
3. Check: Is loss decreasing smoothly? → Good, keep it
4. Check: Is loss oscillating wildly? → LR too high, try 3e-4
5. Check: Is loss barely moving? → LR too low, try 3e-3
6. **For your BRAM net:** LR=1e-3 with cosine decay over 100 epochs is a solid default
7. **For your DDR4 net:** Same, but consider warmup (5 epochs) because the first layer is much larger

### The LR Experiment

One of your first Azure-validated experiments should be LR comparison:
```
Run A: LR = 3e-4 (conservative)
Run B: LR = 1e-3 (default)
Run C: LR = 3e-3 (aggressive)
All else equal. Compare validation loss after 20 epochs.
Pick the winner, use it for all subsequent runs.
```

This takes ~3 hours per run on a T4. Well worth the $1.35 total cost.

---

## 7. Training Cycles and Curriculum — Order Matters

### Why Not Just Throw All Data at It?

You could dump 200M positions into the training loop and let it run for 100 epochs. This works, but you can do better by being strategic about **what data** you train on and **when**.

### Curriculum Learning: Easy → Hard

The idea: start with "easy" positions (clear material advantages, obvious wins/losses) and gradually introduce "hard" ones (equal positions where small positional factors matter).

```
Epochs 1-20:   Positions with |eval| > 200cp (clear advantages)
               → Network learns basic material counting: "rook > bishop"
Epochs 20-50:  All positions
               → Network learns positional nuances
Epochs 50-100: Positions with |eval| < 100cp (balanced positions)
               → Network refines ability to distinguish subtle advantages
```

**Why it works:** Just like teaching a student — start with the fundamentals, then add nuance. If you start with all subtle positions, the network struggles because it doesn't even know that rooks are worth more than pawns yet.

### Self-Play Reinforcement Loop

Once you have a working search engine (Phase 3 of your plan), you can generate training data from your own engine:

```
Cycle 1: Train NNUE on Lichess data (W/L/D)
         → Decent eval, maybe ~2200 ELO
Cycle 2: Play 100K self-play games using NNUE + search
         → Generate positions with YOUR engine's search scores as labels
         → Retrain NNUE on mix: 70% Lichess + 30% self-play
         → Stronger eval, maybe ~2400 ELO
Cycle 3: Play another 100K self-play games with improved NNUE
         → Retrain on mix: 50% Lichess + 50% self-play
         → Even stronger, ~2500 ELO
```

Each cycle improves the NNUE because:
- The search scores from Cycle 2 are higher quality than game outcomes
- The self-play positions are tailored to YOUR engine (it practices on positions it actually reaches)
- This is a form of reinforcement learning — the engine teaches itself

### My Recommendation on W/L/D First

**The reason I recommend W/L/D for initial training is NOT because it's theoretically better.** It's because:

1. **You don't have a search engine yet** (Weeks 1-4 of your plan are infrastructure + BRAM training). You literally can't generate search scores without a search engine.
2. **W/L/D requires zero dependencies.** Download Lichess PGNs → extract positions → label with game result → train. No engine needed.
3. **It establishes your independent baseline.** Everything is yours.
4. **You switch to search scores as soon as you have a search.** Phase 3 builds a minimal alpha-beta, then Phase 4 uses it for self-play data.

**In short: W/L/D first is a practical sequencing decision, not a theoretical one.**

---

## 8. Power-of-2 Layer Sizes and FPGA Memory Allocation

### Short Answer: Yes, Use Powers of 2. Here's Why.

Your intuition is correct but the reason is slightly different than "the FPGA allocates the nearest power-of-2 registers."

### BRAM Allocation

BRAM on the XCK26 comes in **fixed blocks** of 36 kilobits each (configurable as 1×36Kb or 2×18Kb). You can't allocate "just 100 bits of BRAM" — you get a full block.

Each block can be configured as different width × depth combinations:
```
36Kb block configurations:
  32K × 1    (1-bit wide, 32K deep)
  16K × 2
  8K × 4
  4K × 9     ← includes parity bit
  2K × 18
  1K × 36
  512 × 72   ← widest: 72 bits per entry
```

If your layer has 256 neurons with int8 weights, each row of the weight matrix is 256 × 8 = 2,048 bits. To store one row, you need:
- At 36 bits wide: 2048/36 = 57 blocks per row → terrible, wastes space
- At 72 bits wide (512×72 config): 2048/72 = 29 blocks per row → still wastes remainder

**With 256 neurons (power of 2):** The weight storage aligns cleanly with block boundaries. With 300 neurons, you'd waste the space between 300 and 512 (the next power-of-2 depth) in every block.

### DSP Slice Parallelism

The 1,248 DSP slices on the KV260 are your multiply-accumulate (MAC) units. For matrix multiplication, you want to process N multiplications in parallel where N divides evenly into your layer width:

```
Layer width 256:  Process 16 mults in parallel × 16 cycles = done
                  Process 32 mults in parallel × 8 cycles = done
                  Process 64 mults in parallel × 4 cycles = done
                  All clean divisions — no idle DSP slices

Layer width 300:  Process 16 in parallel → 300/16 = 18.75 → need 19 cycles, 4 mults wasted
                  Process 32 in parallel → 300/32 = 9.375 → need 10 cycles, 20 mults wasted
                  Ugly. Wasted hardware on every cycle.
```

### Address Decoding

Power-of-2 sizes mean addresses are just the lower N bits — no modular arithmetic needed:

```
Address for neuron i in a 256-wide layer:  addr = i & 0xFF  (just mask lower 8 bits)
Address for neuron i in a 300-wide layer:  addr = i % 300    (requires a divider — expensive in hardware)
```

### Summary: Your Layer Sizes

| Layer | Size | Power of 2? | Why |
|-------|------|------------|-----|
| Input features | 768 | 768 = 3 × 256 | Not power of 2, but it's sparse (only ~16-32 features active). You never store/process all 768 simultaneously. |
| Accumulator output | 256 | ✅ 2^8 | Clean BRAM storage, clean DSP parallelism |
| Layer 1 input | 512 | ✅ 2^9 | Concat of two 256 accumulators |
| Layer 1 output | 32 | ✅ 2^5 | Small enough that any size works, but 32 is clean |
| Layer 2 output | 32 | ✅ 2^5 | Same |
| Output | 1 | — | Single value, doesn't matter |

**Your current architecture is already all powers of 2.** Good.

For the DDR4 net: 1024 output is 2^10 ✅. The 40,960 input is sparse (only ~30 features active per position), so its non-power-of-2 size doesn't waste hardware.

---

## 9. Memory Hierarchy — BRAM, SRAM, DDR4, VRAM, SSD, MicroSD

Think of memory as a pyramid: faster and more expensive at the top, slower and cheaper at the bottom.

```
                    ┌─────────┐
                    │ Registers│  ← Fastest: single-cycle, ~bytes
                    │ (FPGA)  │     Inside the FPGA fabric itself
                    ├─────────┤
                    │  TCM    │  ← Tightly Coupled Memory (R5F cores)
                    │ (SRAM)  │     128KB, single-cycle, deterministic
                    ├─────────┤
                    │  BRAM   │  ← Block RAM (FPGA on-chip)
                    │         │     648 KB, single-cycle at FPGA clock
                    ├─────────┤
                    │ UltraRAM│  ← On-chip, denser than BRAM
                    │         │     2.25 MB, single-cycle
                    ├─────────┤
                    │  L1/L2  │  ← CPU cache (A53 cores)
                    │  Cache  │     32KB L1 + 1MB L2, 1-10 cycles
                    ├─────────┤
                    │  DDR4   │  ← Main memory (off-chip)
                    │  (RAM)  │     4 GB, ~100ns latency, ~17 GB/s
                    ├─────────┤
                    │  SSD    │  ← Non-volatile storage
                    │ (NVMe)  │     256 GB, ~50-100μs latency
                    ├─────────┤
                    │ MicroSD │  ← Removable flash storage
                    │         │     32 GB, ~1-5ms latency
                    └─────────┘
```

### Detailed Breakdown

#### Registers (FPGA Flip-Flops)
- **What:** Individual bits of storage inside the FPGA logic fabric
- **Speed:** Single clock cycle (5ns at 200MHz)
- **Size:** Thousands available, but each holds just 1 bit
- **Used for:** Pipeline stages, temporary values during computation, state machines
- **Analogy:** Scratch paper on your desk while you calculate

#### SRAM (Static RAM) — includes TCM and CPU caches
- **What:** Fast memory made of 6 transistors per bit (no refresh needed)
- **Speed:** 1-3 clock cycles
- **Size:** Small (KB to low MB) — too expensive per bit for large amounts
- **Used for:** CPU L1/L2 caches, R5F tightly-coupled memory (128KB)
- **Key property:** Deterministic access time — every read takes the same time
- **Analogy:** A small whiteboard next to your desk — fast to read/write, limited space

#### BRAM (Block RAM) — FPGA-specific
- **What:** Dedicated SRAM blocks embedded in the FPGA fabric
- **Speed:** Single FPGA clock cycle (5ns at 200MHz)
- **Size:** 648 KB total on KV260 (144 blocks × 36Kb)
- **Used for:** NNUE weights (your 413KB BRAM net), lookup tables, FIFOs, small buffers
- **Key property:** True dual-port — two independent reads/writes per clock cycle
- **Analogy:** Built-in shelves in your office — fast access, fixed capacity, came with the room
- **Why it matters for you:** Your BRAM-only NNUE lives entirely here = no memory latency during eval

#### UltraRAM — FPGA-specific
- **What:** Higher-density on-chip RAM blocks (288Kb each vs BRAM's 36Kb)
- **Speed:** Single FPGA clock cycle (same as BRAM)
- **Size:** 2.25 MB on KV260 (64 blocks × 288Kb)
- **Used for:** Larger on-chip storage — accumulator state, large lookup tables
- **Key property:** Single-port only (unlike BRAM's dual-port), but 8× denser
- **Analogy:** A filing cabinet in your office — same room (no travel time), holds more, but one person at a time

#### DDR4 (Dynamic RAM)
- **What:** Off-chip memory on separate chips, connected via memory controller
- **Speed:** ~100ns first access (latency), then 17 GB/s burst (throughput)
- **Size:** 4 GB on KV260
- **Used for:** Transposition table, large NNUE weights (DDR4 net), OS memory, buffers
- **Key property:** Must be constantly refreshed (capacitors leak charge) — hence "dynamic"
- **The latency gap:** 100ns = ~20 FPGA clock cycles at 200MHz. This is why putting the accumulator in DDR4 is slower — every move update requires waiting for DDR4.
- **Analogy:** The library down the hall — huge collection, but takes time to walk there

#### VRAM (Video RAM / GPU Memory)
- **What:** GDDR6 memory on your GPU (A2000 has 4GB GDDR6)
- **Speed:** ~500 GB/s bandwidth, ~200-400ns latency
- **Size:** 4 GB on A2000
- **Used for:** Training (PyTorch tensors, gradients, optimizer state all live here during training)
- **Key property:** Optimized for massive parallel access (thousands of GPU cores reading simultaneously)
- **Not on your FPGA board.** Only relevant for training on your desktop. Once training is done, the weights are exported as a binary file and loaded into BRAM/DDR4 on the KV260.
- **Analogy:** A warehouse with 3,000 loading docks — huge throughput when everything moves in parallel

#### SSD (Solid-State Drive)
- **What:** NAND flash memory with a controller, non-volatile (keeps data when powered off)
- **Speed:** Sequential: ~3 GB/s (NVMe). Random 4KB reads: ~50-100μs per read
- **Size:** Your 256GB NVMe
- **Used for:** Syzygy tablebases (149GB), persistent storage
- **Key property:** Non-volatile! Data survives power cycles. But random access is 1000× slower than DDR4.
- **Why NOT for TT:** A transposition table needs millions of random reads per second. At 50μs per random read, SSD gives ~20K reads/sec. You need ~10M reads/sec. Off by 500×.
- **Why FINE for Syzygy:** Syzygy is probed ~100 times per game, not millions. 50μs × 100 = 5ms total. Invisible.
- **Analogy:** A storage unit across town — huge capacity, takes a trip to get there, but fine for things you rarely need

#### MicroSD Card
- **What:** Tiny flash storage card, similar technology to SSD but slower controller
- **Speed:** ~100 MB/s sequential (Class 10 / UHS-I). Random reads: ~1-5ms
- **Size:** Your 32GB card
- **Used for:** Boot image (PetaLinux), engine binary, NNUE weight files, opening book, config
- **Key property:** Removable, cheap, adequate for one-time boot-up reads
- **Analogy:** A USB stick in your pocket — convenient, slow, good enough for carrying files around

### How This Maps to Your Project

```
┌─────────────────────────────────────────────────────┐
│ BRAM + UltraRAM (3.3 MB, single-cycle)              │
│   → NNUE weights (413 KB)                           │
│   → Accumulator state (512 bytes × max_depth)        │
│   → Layer intermediate buffers                       │
├─────────────────────────────────────────────────────┤
│ DDR4 (4 GB, ~100ns latency)                          │
│   → Transposition table (64-256 MB)                  │
│   → Search stack, move lists                         │
│   → DDR4 NNUE accumulator weights (40 MB) [large net]│
│   → Linux OS memory                                  │
├─────────────────────────────────────────────────────┤
│ SSD via USB 3.0 (256 GB, ~50-100μs)                  │
│   → Syzygy endgame tablebases (149 GB)               │
│   → Deep TT archive (optional)                       │
├─────────────────────────────────────────────────────┤
│ MicroSD (32 GB, ~1-5ms)                              │
│   → Boot image, engine binary, NNUE weight files     │
│   → Opening book (~5-50 MB)                          │
│   → Logs, config                                     │
└─────────────────────────────────────────────────────┘
```

---

## 10. Design Choices Explained

### Why Bitboards?

A chess board has 64 squares. A 64-bit integer has 64 bits. Each bit represents one square. One integer = one "layer" of the board.

```
White pawns:     0x000000000000FF00  (rank 2 starting position)
Black knights:   0x4200000000000000  (b8 and g8)
All white pieces: white_pawns | white_knights | white_bishops | ...
```

**Why this is better than an 8×8 array:**
- Move generation becomes bitwise operations (AND, OR, SHIFT) — single CPU instructions
- "All squares attacked by white bishops" is one operation, not a loop
- 12 bitboards (one per piece type × color) = 96 bytes total board state
- ARM A53 has native 64-bit operations — one instruction processes all 64 squares

### Why int8 Quantization?

During training, weights are float32 (32 bits each). For FPGA inference, you convert to int8 (8 bits):

```
Float32 weight:  0.0372941...  (32 bits = 4 bytes)
Int8 weight:     9              (8 bits = 1 byte, representing 9/256 ≈ 0.0352)
```

**Why:**
- **4× less memory:** 413KB instead of 1.65MB — fits in BRAM
- **Simpler hardware:** 8-bit multipliers use ~4× fewer LUTs and DSPs than 32-bit
- **Faster:** More operations per clock cycle
- **Minimal quality loss:** With quantization-aware training (QAT), the network learns to be robust to rounding. Typical loss: 10-30 ELO (1-3% of total strength)

### Why ClippedReLU?

```
ClippedReLU(x) = max(0, min(127, x))

        127 ─────────────────
           /
          /
         /
────────0
```

**Why not regular ReLU?** Regular ReLU has unbounded output (can be 0 to +infinity). With int8 arithmetic, values above 127 overflow and wrap around (255 → -1 in signed int8). ClippedReLU caps at 127, preventing overflow.

**Why it's hardware-friendly:** It's just two comparisons and a mux (multiplexer). No division, no exponentiation, no lookup tables. In FPGA, it's literally:
```verilog
assign output = (input < 0) ? 8'd0 : (input > 127) ? 8'd127 : input[7:0];
```
One LUT, one clock cycle.

### Why Two Perspectives (White + Black)?

The NNUE evaluates from BOTH sides and concatenates:

```
White's view: "I see my knight on f3, opponent's pawn on e5..."
  → accumulator_white (256 values)

Black's view: "I see my pawn on e5, opponent's knight on f3..."
  → accumulator_black (256 values)

Concatenate: [accumulator_white | accumulator_black] = 512 values
  → Feed into Layer 1
```

**Why not just one perspective?** Chess is NOT symmetric. White having a knight on f3 means something different than Black having a knight on f3 (proximity to each king, pawn structures, etc.). Two perspectives lets the network learn asymmetric patterns.

### Why Zobrist Hashing (for TT)?

To look up a position in the transposition table, you need a hash. Zobrist hashing:

```
hash = 0
for each piece on each square:
    hash ^= random_number[piece_type][square]
```

Where `random_number` is a precomputed table of 64-bit random values (12 piece types × 64 squares = 768 random numbers).

**Why XOR?** Because it's **incrementally updatable** (same insight as NNUE!):
```
Before move e2→e4:  hash = ... ^ random[white_pawn][e2] ^ ...
After move e2→e4:   hash ^= random[white_pawn][e2]  // remove pawn from e2
                    hash ^= random[white_pawn][e4]  // add pawn to e4
```
Two XOR operations per move, regardless of board complexity.

---

## 11. Your Testing Methodology — The Right Approach

Your instinct is exactly right. Here's the framework:

### The Three Engine Configurations

```
Config A: Alpha-Beta + Material Eval (no NNUE)
          → Baseline. How strong is your search alone?

Config B: Alpha-Beta + BRAM NNUE (small net)
          → How much does the small NNUE improve over material?

Config C: Alpha-Beta + DDR4 NNUE (large net)
          → How much does the large net add over the small?
```

### Test Matrix

| Matchup | What It Measures | Expected Outcome |
|---------|-----------------|------------------|
| A vs A (self-play) | Sanity check — should be ~50% | ~50/50 |
| B vs A | NNUE value over material eval | B wins ~70-80% |
| C vs A | Large NNUE value over material | C wins ~85-95% |
| C vs B | Marginal value of large net | C wins ~55-65% |
| A vs Stockfish depth 5 | Calibrate your ELO | Stockfish wins ~80% |
| B vs Stockfish depth 8 | Calibrate BRAM net ELO | Varies |
| C vs Stockfish depth 10 | Calibrate large net ELO | Varies |

### Platform Comparison

Run each config on both platforms to measure the hardware impact:

```
Desktop (i7 + A2000, 32GB):
  A-desktop, B-desktop, C-desktop

KV260 (ARM A53, FPGA, 4GB DDR4):
  A-kv260, B-kv260, C-kv260

Matchups:
  A-desktop vs A-kv260    → measures CPU speed impact on search-only engine
  B-desktop vs B-kv260    → measures CPU speed impact WITH NNUE
  B-kv260 vs B-kv260-fpga → measures FPGA acceleration value (CPU eval vs FPGA eval)
```

### Tournament Parameters

```bash
cutechess-cli \
  -engine name="NoNNUE" cmd="./engine" arg="--eval=material" \
  -engine name="BRAMNet" cmd="./engine" arg="--eval=bram_nnue" \
  -each proto=uci tc=10+0.1 \    # 10 seconds + 0.1s increment
  -rounds 500 \                   # 500 game pairs (1000 games total)
  -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05 \  # stop early if result is clear
  -openings file=openings.pgn format=pgn order=random \
  -pgnout results.pgn
```

### What Goes in Your Report

For each comparison, record:
- Win/Draw/Loss ratio
- ELO difference (±confidence interval)
- Average search depth reached
- Average nodes per second
- Average time per move
- Notable games (blunders, brilliant moves)

This data is pure gold for your admissions portfolio and technical paper.

---

## 12. Implementation Steps — From Zero to Working NNUE

Given your background (basic ML class + strong Connect Four engine), here's the order to learn and build:

### Step 1: Understand the Data (Week 1, Day 1-2)

Before writing any training code, understand your input:

```
1. Download ONE month of Lichess data (~2-5 GB compressed)
   https://database.lichess.org

2. Decompress: zstd -d lichess_db_standard_rated_2026-01.pgn.zst

3. Open in a text editor. A PGN game looks like:
   [Event "Rated Blitz game"]
   [White "player1"]
   [Black "player2"]
   [Result "1-0"]
   [WhiteElo "2100"]
   [BlackElo "2050"]
   
   1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 ... 1-0

4. Your parser needs to:
   - Read the Result tag → W/L/D label
   - Read the WhiteElo/BlackElo → filter for 2000+
   - Replay the moves → at each position, extract the 768 features
   - Save as binary: [features_white, features_black, label]
```

### Step 2: Build Feature Encoding (Week 1, Day 2-4)

The 768-feature encoding for BRAM net:

```
768 = 12 piece types × 64 squares

Piece types (from white's perspective):
  0: White Pawn      6: Black Pawn
  1: White Knight     7: Black Knight
  2: White Bishop     8: Black Bishop
  3: White Rook       9: Black Rook
  4: White Queen     10: Black Queen
  5: White King      11: Black King

Feature index = piece_type × 64 + square_index

Example: White knight on f3 (square 21)
  feature_index = 1 × 64 + 21 = 85
  features[85] = 1  (all other features = 0)
```

A position typically has ~16-32 active features (one per piece on the board).

**Unit test:** Encode the starting position. Verify exactly 32 features are active (16 white pieces + 16 black pieces at their starting squares).

### Step 3: Build the PyTorch Model (Week 1, Day 4-5)

```python
import torch
import torch.nn as nn

class ChessNNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # SHARED accumulator. EmbeddingBag pulls only the active feature rows
        # and sums them — vastly faster than a dense matmul on a sparse vector.
        # See §13.2 for why this is critical.
        self.accumulator = nn.EmbeddingBag(
            num_embeddings=768,
            embedding_dim=256,
            mode='sum',
        )
        self.acc_bias = nn.Parameter(torch.zeros(256))
        
        self.l1 = nn.Linear(512, 32)
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)
    
    def clipped_relu(self, x):
        # [0, 1] in float32 maps to [0, 127] after int8 quantization.
        return torch.clamp(x, 0.0, 1.0)
    
    def forward(self, stm_feats, stm_offsets, nstm_feats, nstm_offsets):
        """
        stm  = side-to-move perspective
        nstm = not-side-to-move perspective
        Both are sparse: pass flat tensors of active feature indices + batch offsets.
        See §13.2 for the input format.
        """
        stm_acc  = self.clipped_relu(self.accumulator(stm_feats,  stm_offsets)  + self.acc_bias)
        nstm_acc = self.clipped_relu(self.accumulator(nstm_feats, nstm_offsets) + self.acc_bias)
        
        # CRITICAL: concat order is [stm, nstm], NOT [white, black].
        # The network needs to know whose turn it is, encoded by position.
        # See §13.1 for why this matters — getting it wrong costs 200-400 ELO.
        x = torch.cat([stm_acc, nstm_acc], dim=1)  # (batch, 512)
        
        x = self.clipped_relu(self.l1(x))
        x = self.clipped_relu(self.l2(x))
        x = self.l3(x)
        
        # Raw output is centipawn-scaled from STM's perspective.
        # For BCE training: wrap in sigmoid(x / 400) — see §13.5.
        # For inference: use directly as centipawn score.
        return x.squeeze(-1)
```

**Key details:**
- The `accumulator` is a single shared layer applied to both perspectives — halves parameter count vs a non-shared `1536→256` design.
- Side-to-move concatenation ordering is enforced at the call site (your DataLoader passes `stm`/`nstm` tensors already in the right order).
- The network output is raw centipawn-scaled; the sigmoid is applied at the loss layer, not inside the model. This keeps inference simple.

### Step 4: Build the Training Loop (Week 1-2)

```python
model = ChessNNUE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
TOTAL_EPOCHS = 100
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)

QAT_START_EPOCH = int(TOTAL_EPOCHS * 0.5)  # see §13.7
SIGMOID_SCALE = 400.0                       # see §13.5

for epoch in range(TOTAL_EPOCHS):
    if epoch == QAT_START_EPOCH:
        enable_quantization_aware_training(model)  # fake-quantize hooks
    
    model.train()
    for batch in train_loader:
        # Sparse-feature batch from your DataLoader. See §13.2 for format.
        stm_feats, stm_offsets, nstm_feats, nstm_offsets, target_wld = batch
        # target_wld: 1.0 = STM wins, 0.5 = draw, 0.0 = STM loses
        
        raw_score = model(stm_feats, stm_offsets, nstm_feats, nstm_offsets)
        pred_wp   = torch.sigmoid(raw_score / SIGMOID_SCALE)
        loss      = nn.BCELoss()(pred_wp, target_wld)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate_on_validation_set(model, val_loader)
    
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    # Save checkpoint every 5 epochs (Azure spot preemption protection)
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f'checkpoint_epoch_{epoch}.pt')
```

**Important notes:**
- Your DataLoader is responsible for emitting `(stm_feats, stm_offsets, nstm_feats, nstm_offsets, target)` tuples. See §13.2 for the sparse-feature format and §13.4 for which positions to actually include.
- `SIGMOID_SCALE = 400.0` matches Stockfish's centipawn-to-win-probability conversion. If you change it, change it everywhere — see §13.5.
- The check `if epoch == QAT_START_EPOCH` is the QAT switch. Without QAT, expect 50-100 ELO loss on int8 export. See §13.7.

### Step 5: Export to int8 (After Training)

```python
def export_int8(model, output_path):
    """Export trained model weights as int8 binary for FPGA loading."""
    with open(output_path, 'wb') as f:
        for name, param in model.named_parameters():
            # Scale float weights to int8 range [-128, 127]
            scale = 127.0 / param.abs().max()
            quantized = (param * scale).round().clamp(-128, 127).to(torch.int8)
            
            # Write scale factor (float32) then quantized weights (int8)
            f.write(struct.pack('f', scale.item()))
            f.write(quantized.numpy().tobytes())
            
            print(f"{name}: shape={list(param.shape)}, scale={scale:.4f}, "
                  f"max_error={((param - quantized/scale).abs().max()):.6f}")
```

### Step 6: Build a Minimal Search to Test (Week 4-5)

```
1. Implement board representation (bitboards — port from Connect Four concepts)
2. Implement move generation (legal moves only — test with perft)
3. Implement basic alpha-beta:
   - Iterative deepening to depth 6-8
   - Material eval first (pawn=100, knight=320, bishop=330, rook=500, queen=900)
   - Test: does it play legal chess? Does it capture free pieces?
4. Replace material eval with NNUE:
   - Load int8 weights at startup
   - Implement forward pass in C++ (just matrix multiply + clamp)
   - Maintain accumulator stack (push on make_move, pop on unmake_move)
5. Implement UCI protocol (so cutechess-cli can talk to it)
6. Run tournaments!
```

### Step 7: Iterate

```
Train NNUE → Test in engine → Analyze weaknesses → Adjust training → Repeat
```

This loop is where the real learning happens. Your first NNUE will be mediocre. Your fifth will be strong.

---

## 13. Common Pipeline Pitfalls (Read Before Building)

These are the correctness and performance issues that bite first-time NNUE implementers. **Address them BEFORE your first training run**, not after you discover them three weeks in.

### 13.1 Side-to-Move Ordering (Correctness Bug)

When concatenating the two accumulators, the order **must depend on whose turn it is**, not be fixed as [white, black]:

```python
# WRONG — network has no way to know whose turn it is
x = torch.cat([white_acc, black_acc], dim=1)

# RIGHT — network always sees "my perspective first"
if side_to_move == WHITE:
    x = torch.cat([white_acc, black_acc], dim=1)
else:
    x = torch.cat([black_acc, white_acc], dim=1)

# Equivalent and cleaner — track perspectives explicitly in your data pipeline:
x = torch.cat([stm_acc, nstm_acc], dim=1)
```

This is why the convention in NNUE literature is `stm` (side-to-move) and `nstm` (not-side-to-move), not "white" and "black."

The network's output is "centipawns from STM's perspective" — positive means good for whoever is about to move. Your search code expects this (negamax flips signs at each ply).

**Without this fix:** The network has no way to distinguish "white to move, position X" from "black to move, mirrored position X." Predictions become a blurred average across both cases. Expect **200-400 ELO worse** than a correct implementation. This is the single most common from-scratch NNUE bug.

### 13.2 Use EmbeddingBag for Sparse Features (10-30× Training Speedup)

A position has 16-32 active features out of 768. Using `nn.Linear(768, 256)` does a dense 768×256 matmul where 95%+ of the input is zero — pure waste.

Replace with `nn.EmbeddingBag`, which pulls only the active rows and sums them:

```python
self.accumulator = nn.EmbeddingBag(
    num_embeddings=768,
    embedding_dim=256,
    mode='sum',
)
self.acc_bias = nn.Parameter(torch.zeros(256))

# Inputs are now (active_feature_indices, batch_offsets), not dense vectors
stm_acc = self.accumulator(stm_feat_idx, stm_offsets) + self.acc_bias
```

**Speedup: 10-30×** on training throughput depending on hardware. This is the difference between "200M positions in 12 hours" and "200M positions in 6 days."

Your DataLoader must produce sparse inputs. Format:
- `stm_feat_idx`: flat 1D tensor of all active feature indices for the batch
- `stm_offsets`: 1D tensor — index where each sample starts in `stm_feat_idx`

Example: batch of 3 positions with 16, 18, 20 active features each:
```
stm_feat_idx = [f00, f01, ..., f15,  f10, f11, ..., f17,  f20, f21, ..., f19]
                ←   position 0   →   ←   position 1   →   ←   position 2   →
stm_offsets  = [0, 16, 34]   # cumulative starts: 0, 0+16, 0+16+18
```

### 13.3 Accumulator MUST Be int16, Not int8 (Hardware Correctness)

When the FPGA computes the accumulator output from int8 weights:

```
accumulator[i] = sum over active features f of W[f][i]
```

With 32 active features and int8 weights in [-128, 127], the sum can reach **±4,064** — far outside int8 range. **The accumulator MUST be stored and updated as int16** (or int32 for safety margin).

Apply ClippedReLU to convert int16 → uint8 before feeding into Layer 1:

```
acc_int16 = sum of int8 weight rows                  // range: ~±4,000
acc_uint8 = max(0, min(127, acc_int16 >> shift))     // range: 0-127
```

The `>>` shift is your **quantization scale** — chosen so training-time activations rarely saturate at 127. Default starting point: `shift = 6` (divides by 64). Tune by measuring how often activations clip during training.

**If you skip this** and store the accumulator as int8, overflow wraps silently and the network produces garbage scores on certain positions. This is the most common FPGA NNUE bug after side-to-move ordering.

### 13.4 Position Filtering (Training Quality)

Naively extracting all positions from Lichess PGNs gives biased data:
- The starting position appears in every game (massively overrepresented)
- Common opening positions are over-sampled by orders of magnitude
- Positions in already-decided games (one side blundered 20 moves ago) carry wrong labels — the loser's "best moves" are actually losing because the game is already lost
- Positions reached in time-trouble carry noisy labels (random blunders, not real evaluations)

Minimum filtering before training:

```python
def should_use_position(board, ply, game, rng):
    # Skip opening book region
    if ply < 16:
        return False
    # Skip positions where one side already has overwhelming material
    if abs(material_balance(board)) > 500:  # > 5 pawn-equivalents
        return False
    # Skip positions where the next move is a capture — these belong in
    # quiescence training, not the static eval set
    if board.is_capture_next():
        return False
    # Skip games decided by timeout / disconnection (noisy labels)
    if game.termination in ('time forfeit', 'abandoned', 'rules infraction'):
        return False
    # Sample 1-in-8 to reduce redundancy between consecutive plies
    if rng.random() > 0.125:
        return False
    return True
```

This drops your raw 1B PGN positions to ~100-150M training-suitable positions. **Counter-intuitively, 100M filtered positions trains a stronger net than 1B unfiltered positions** because the loss signal is cleaner and the network doesn't waste capacity memorizing common opening lines.

### 13.5 Sigmoid Scaling Constant (Numerical Consistency)

The standard NNUE convention is to map between centipawns and win-probability via the sigmoid:

```python
# Stockfish convention: 400 centipawns ≈ 73% win probability
SIGMOID_SCALE = 400.0

# During training:
target_wp     = game_result                                # 1.0 win / 0.5 draw / 0.0 loss
# OR if using SF eval targets:
# target_wp   = torch.sigmoid(stockfish_score / SIGMOID_SCALE)
predicted_wp  = torch.sigmoid(network_output / SIGMOID_SCALE)
loss          = nn.BCELoss()(predicted_wp, target_wp)
```

**Pick your scaling factor (default 400) and use it EVERYWHERE** — training loss, inference output interpretation, and FPGA quantization bit-width planning. If your search engine expects "score in centipawns" but your NNUE outputs "raw network value scaled by 1024", your alpha-beta windows and aspiration intervals will all be the wrong size and the engine will mysteriously play worse than its eval suggests.

### 13.6 Horizontal Symmetry Augmentation (Free 2× Data)

Chess is approximately horizontally symmetric (excluding castling rights and en passant). Mirroring positions left-right essentially doubles your dataset for free:

```python
def augment_horizontal(features, score):
    # Mirror file: a-file ↔ h-file, b ↔ g, c ↔ f, d ↔ e
    mirrored = mirror_files(features)
    return [(features, score), (mirrored, score)]
```

Skip augmentation for positions where castling rights still exist or en passant is available — those aren't truly symmetric. For a typical Lichess dataset, ~70% of positions qualify for free mirroring.

Vertical mirroring (swap white ↔ black, flip side-to-move, negate score) also works but is redundant if your network already processes both perspectives correctly.

### 13.7 Quantization-Aware Training (QAT)

Train in float32 for the first 50-70% of epochs (fast convergence), then enable QAT for the final 30-50% so the network learns to be robust to int8 rounding:

```python
QAT_START_EPOCH = int(TOTAL_EPOCHS * 0.5)

for epoch in range(TOTAL_EPOCHS):
    if epoch == QAT_START_EPOCH:
        # Insert fake-quantize ops into the forward pass
        # (you may need a custom implementation for EmbeddingBag)
        enable_fake_quantize(model)
    # ... rest of training loop ...
```

**Without QAT:** Expect to lose **50-100 ELO** when exporting to int8 because the float32-trained weights have patterns that don't survive rounding.

**With QAT:** Loss is typically **< 20 ELO** because the network learns weight distributions that quantize well.

PyTorch's built-in `torch.quantization` framework doesn't fully support `EmbeddingBag`, so you'll likely need to write the fake-quantize op inline:

```python
class FakeQuantize:
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
    
    def __call__(self, x):
        scale = (2 ** (self.n_bits - 1) - 1) / x.abs().max().clamp(min=1e-8)
        return torch.round(x * scale) / scale  # round-trip through int range
```

Apply this to weights during the forward pass during the QAT phase.

### 13.8 Pre-Flight Checklist (Before Your First Training Run)

```
[ ] Side-to-move concatenation uses stm/nstm ordering, NOT fixed white/black
[ ] Accumulator output is int16 on hardware (float32 during training, with awareness of int16 target)
[ ] Sparse features encoded via EmbeddingBag (not dense Linear) — 10-30× speedup
[ ] Position filtering applied during data extraction (skip ply<16, |mat|>500, capture-next, etc.)
[ ] Sigmoid scaling factor (default 400) documented and used consistently across all code
[ ] Horizontal mirroring augmentation applied where valid (no castling/en-passant)
[ ] QAT enabled for the final 30-50% of epochs
[ ] Bit-accurate PyTorch reference model exists (will be needed for FPGA verification later)
[ ] 5-epoch dry run on 10M positions completed; loss decreases monotonically
[ ] Checkpoint save/resume tested by killing and restarting mid-epoch
```

If you can check all of these, you're set up to train a clean, exportable NNUE. If you skip them, you'll spend weeks debugging mysterious ELO regressions and quantization mismatches.

---

*Guide created May 28, 2026 for the Chess NNUE on FPGA project.*
*Updated May 28, 2026 (afternoon): added §13 covering side-to-move ordering, sparse-feature handling, int16 accumulator requirement, position filtering, sigmoid scaling, symmetry augmentation, and QAT; rewrote §12 Step 3 PyTorch model to use EmbeddingBag and STM-first concat; updated §12 Step 4 training loop with QAT phase and explicit sigmoid scaling; clarified shared-accumulator structure and ~213K parameter count in §1.*

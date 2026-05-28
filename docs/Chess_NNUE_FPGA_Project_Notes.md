# Chess NNUE on FPGA — Project Reference Document

**Date:** May 27, 2026
**Goal:** Build a grandmaster-level chess engine (~3000 ELO target) with NNUE evaluation on FPGA hardware, entirely from scratch — no Stockfish or external engine code. Use Microsoft internship resources (Azure credits, 3D printers) and present as a portfolio project for Master's admissions at Stanford, Harvard, MIT, and CMU (CS + EE).

---

## Table of Contents

1. [Project Overview & Goals](#1-project-overview--goals)
2. [NNUE Architecture](#2-nnue-architecture)
3. [FPGA Hardware — Kria KV260](#3-fpga-hardware--kria-kv260)
4. [NNUE Design: BRAM-Only vs DDR4 Accumulator](#4-nnue-design-bram-only-vs-ddr4-accumulator)
5. [Storage Architecture](#5-storage-architecture)
6. [Minimax / Search Engine Design](#6-minimax--search-engine-design)
7. [Training Pipeline (Technical Reference)](#7-training-pipeline)
8. [Azure Budget Strategy](#8-azure-budget-strategy)
9. [Testing & ELO Estimation](#9-testing--elo-estimation)
10. [Connect Four Engine — Existing Codebase & Chess Porting Guide](#10-connect-four-engine--existing-codebase--chess-porting-guide)
11. [Parts List & Purchase Links](#11-parts-list--purchase-links)
12. [Admissions Strategy](#12-admissions-strategy)
13. [Key References & Resources](#13-key-references--resources)
14. [Revised NNUE Training Plan (10-12 Weeks, ~$450 Azure)](#14-revised-nnue-training-plan-10-12-weeks-450-azure)
15. [Post-Training Work (After Internship)](#15-post-training-work-after-internship)
16. [Decisions Made & Rationale](#16-decisions-made--rationale)
17. [Risks & Mitigations](#17-risks--mitigations)
18. [Alternative Hardware Considered](#18-alternative-hardware-considered)
19. [Cortex-R5F Real-Time Cores — Architecture & Usage](#19-cortex-r5f-real-time-cores--architecture--usage)
20. [FPGA Inference Pipeline — Architecture & Implementation](#20-fpga-inference-pipeline--architecture--implementation)

---

## 1. Project Overview & Goals

- Build a complete chess engine from scratch (move generation, search, evaluation, UI)
- Train a custom NNUE (Efficiently Updatable Neural Network) for position evaluation
- Deploy NNUE inference on FPGA for hardware-accelerated evaluation
- Physical device with touchscreen for interactive play
- 3D-printed case (Microsoft 3D printers)
- Target ELO: ~3000 (realistic: 2200–2900 depending on time invested)
- Budget: ~$436 hardware + ~$450 Azure credits ($150/month × 3 months, free intern benefit)
- Timeline: 10-12 weeks (training focus), then post-internship for hardware integration

---

## 2. NNUE Architecture

### What is NNUE?
Efficiently Updatable Neural Network — pioneered in Shogi, adopted by Stockfish. A shallow network (2-4 layers) designed for fast incremental evaluation. Key insight: only a few input features change per move, so the first layer ("accumulator") can be updated incrementally rather than recomputed.

### Stockfish NNUE Reference (HalfKAv2)

> **Note:** Stockfish's NNUE architecture has evolved significantly since 2020. The below reflects the general HalfKAv2 concept. Current Stockfish (2024+) uses king-bucket features and a different layer structure. Since you're building from scratch, your architecture doesn't need to match — this is just a reference point for scale.

```
Input:   ~49,152 features (HalfKAv2: king-piece-square)
Layer 0: 49,152 → 1024 (accumulator) — ~50MB in int8
Layer 1: 1024 → 8                    — ~8KB
Layer 2: 8 → 32                      — ~256B
Layer 3: 32 → 1                      — ~32B
Activation: ClippedReLU (hardware-friendly — just a clamp)
Output: centipawn evaluation score
Quantized to int8/int16
```

---

## 3. FPGA Hardware — Kria KV260

### KV260 vs KR260
Both use the **identical XCK26 SoC** (Zynq UltraScale+). Same FPGA fabric. Difference is carrier board I/O only.

| | KV260 (~$280) | KR260 (~$350) |
|---|---|---|
| Focus | Vision/AI | Robotics/Industrial |
| Extra I/O | MIPI CSI, HDMI | Dual GbE w/ TSN, CAN, industrial |
| For chess? | Sufficient | Only if adding a robot arm |

**Decision: KV260** — save the $70.

### KV260 Exact FPGA Resources

| Resource | Count | Usable Storage |
|----------|-------|----------------|
| **BRAM** | 144 blocks × 36Kb (kilobits) | **648 KB** |
| **UltraRAM** | 64 blocks × 288Kb (kilobits) | **2,304 KB (2.25 MB)** |
| **Total on-chip SRAM** | — | **~3.3 MB** |
| **DSP slices** | 1,248 | Matrix multiply units |
| **Logic cells** | 256K | Control logic, pipelines |
| **DDR4** | 4 GB | Off-chip, high bandwidth (~17 GB/s) |

### CPU Resources
- Quad ARM Cortex-A53 @ 1.3GHz (application processing — runs Linux, search engine, UCI)
- Dual ARM Cortex-R5F (real-time cores — see [Section 19](#19-cortex-r5f-real-time-cores--architecture--usage) for detailed architecture and usage ideas)

### Architecture Split
- **CPU/ARM** → Minimax/Alpha-Beta search, move generation, transposition table
- **FPGA** → NNUE forward pass (matrix multiply, ClippedReLU, accumulator update)

---

## 4. NNUE Design: BRAM-Only vs DDR4 Accumulator

### Option A: BRAM-Only (Recommended to build first)

**Standard NNUE design with a shared-weight accumulator.** The same 768→256 linear layer is applied independently to each perspective's sparse features. This is what Stockfish, Leela's NNUE forks, and nnue-pytorch all do.

```
Feature encoding:  768 features (12 piece types × 64 squares) per perspective
                   Sparse binary inputs (~16-32 active out of 768)
                   Two perspectives evaluated with the SAME accumulator weights

Layer 0 (Accumulator, SHARED): 768 → 256   weights: 192 KB (int8, used by both sides)
                               Output:     2 × 256 int16 (stm + nstm accumulators)
                               After ClippedReLU + concat (STM-first): 512 uint8
Layer 1:                       512 → 32    weights: 16 KB
Layer 2:                       32 → 32     weights: 1 KB
Layer 3:                       32 → 1      weights: 32 bytes
────────────────────────────────────
TOTAL weight storage: ~210 KB — fits in BRAM with ~3.1 MB to spare
Total parameters:     ~213K (shared accumulator counted once)
```

> **Note on shared vs non-shared accumulator:** Earlier drafts of this doc framed the accumulator as `1,536 → 256` with separate weights per perspective (~410K params, ~384 KB). That's a valid design but doubles parameter count for marginal benefit. The shared-weight version is the standard NNUE approach and is what this project uses. The trick that makes it work is **concatenating in side-to-move order** — see [Deep Dive Guide §13.1](./NNUE_Deep_Dive_Guide.md#131-side-to-move-ordering-correctness-bug).

- Everything on-chip = single-cycle access, no DDR4 latency
- ~213K parameters (shared accumulator)
- Simpler FPGA design
- Target: 5-20M evals/sec (depends on pipeline depth and clock; 200MHz with deep pipelining needed for higher end)
- Expected eval strength: ~2400-2600 ELO

**Scaling option (still BRAM):**
```
Layer 0: 1,536 → 512 = 786 KB → uses BRAM + UltraRAM (still fits)
```
Or HalfKP-lite (king-relative, 6,144 features):
```
Layer 0: 6,144 → 256 = 1.5 MB → fits in BRAM + UltraRAM
```

### Option B: DDR4 Accumulator (Build second, use Azure credits)

```
DDR4 (4GB):   Layer 0 weights — HalfKP 40,960 × 1024 = ~40 MB (int8)
BRAM (3.3MB): Accumulator state (2 × 1024 × int16 = 4 KB)
              Layer 1-3 weights (~9 KB)
              Buffers, pipeline registers
FPGA logic:   All MAC operations, ClippedReLU, accumulator update
```

- Full Stockfish-scale architecture
- ~40M parameters
- More complex FPGA design (AXI DMA controller, burst reads, prefetch buffer)
- Target: 5-15M evals/sec
- Expected eval strength: ~2800-3000 ELO

### Comparison

| Factor | BRAM-only (768→256) | DDR4 accumulator (40K→1024) |
|---|---|---|
| Eval quality | ~2400-2600 ELO | ~2800-3000 ELO |
| Eval speed | Fast (~5-20M eval/s) | Slower (~5-15M eval/s) |
| Training data needed | 200M positions | 1B+ positions |
| Training time | ~8-15 hours (local A2000) | ~50-100 hours (Azure) |
| FPGA complexity | Moderate | High |
| Overall ELO | ~2200-2600 | ~2500-2900 |

### How DDR4 Changes Training
- Training process in PyTorch is **almost identical** — PyTorch doesn't care where weights live on hardware
- **More data needed** — 40M params vs 200K params means proportionally more data to avoid overfitting
- **Longer training** — ~30-60 min per epoch vs ~2 min per epoch
- **More complex feature encoding** — HalfKP requires king-position × piece-square mapping
- **FPGA design is harder** — need DMA, caching, handle DDR4 latency

### Key Insight: Incremental Update
Each move only changes 2-4 feature rows out of 40K, so DDR4 reads are only 2-8 KB per update. At 17 GB/s bandwidth, that's ~200ns — fast, but not single-cycle like BRAM.

### Strategy Decision
**Build BRAM-only first, then DDR4 version. A/B comparison = publishable result.**

---

## 5. Storage Architecture

### SSD (256GB NVMe via USB 3.0)
- **Syzygy endgame tablebases** — 6-piece tables ~150GB (7-piece = ~140TB, skip it)
- **Persistent deep TT entries** (depth ≥ 20) as endgame knowledge cache
- **Opening book learning** — save positions with surprising evals

**USB 3.0 performance for Syzygy — not a bottleneck:**
- Syzygy probes are infrequent (~1-10 KB per lookup, only at ≤6 pieces remaining)
- USB 3.0 provides ~500 MB/s throughput, ~1-2ms latency per probe
- You probe a few hundred times per game, not millions — latency is invisible vs search time
- NVMe over USB 3.0 is no faster than SATA over USB 3.0 for this use case (both bottlenecked by USB)
- **Verdict: Stick with NVMe + USB enclosure** — $55 total, future-proof if you upgrade to M.2 board later

**Where USB 3.0 would NOT work:** Storing the transposition table on SSD (millions of reads/sec). TT stays in DDR4 RAM. This is a non-issue with the current design.

### 32GB MicroSD
- PetaLinux or Ubuntu boot image
- Chess engine binary
- Opening book (Polyglot format, ~few MB)
- NNUE weight files
- Logs, config files

### Transposition Table
- Primary TT: 64MB+ in DDR4 (in-memory, fast)
- Zobrist hashing with depth-preferred replacement scheme
- Persistent deep entries on SSD — useful but limited value (entries are depth-specific)
- Better: persistent **opening book** from self-play discoveries

---

## 6. Minimax / Search Engine Design

### Porting from Connect Four to Chess
Key upgrades needed:
1. **Alpha-Beta with move ordering** (likely already have from Connect Four)
2. **Transposition table** — Zobrist hashing, much larger state space
3. **Iterative deepening** with aspiration windows
4. **Null move pruning, Late Move Reductions (LMR)**
5. **Quiescence search** — search captures/checks at leaf nodes (critical for chess)
6. **Move generation** — bitboard representation (64-bit), magic bitboards for sliding pieces
7. **Multi-threaded search** — Lazy SMP across 4 ARM cores

### Board Representation
- 64-bit bitboards for each piece type (12 bitboards)
- Magic bitboards for sliding piece move generation
- Maps well to hardware (64-bit operations native on ARM A53)

### For ~3000 ELO
Need depth 20+ with good pruning and a strong eval.

---

## 7. Training Pipeline

### Training Strategy
- **BRAM-only net (768→256):** Train locally on A2000 4GB GPU
- **DDR4 net (40K→1024):** Train on Azure with $150 credits
- Train BRAM net first to learn the pipeline, then scale up

### A2000 for BRAM-Only Net
| Metric | Value |
|--------|-------|
| VRAM needed | ~300-500 MB |
| Available VRAM | 4 GB — plenty |
| Batch size | 8,192–16,384 |
| Training time (200M positions, 100 epochs) | ~8-15 hours |
| Cost | $0 |

### Step 1: Generate Training Data

**Option A — Lichess data with game outcomes (recommended to start):**
- Download from https://database.lichess.org
- Extract positions, label with game result: +1/0/-1
- 100M positions minimum, 500M+ ideal
- Write your own PGN parser

**Option B — Self-play bootstrap (fully self-contained):**
1. Start with random NNUE + minimax at depth 4-6
2. Play self-play games, record positions + search scores
3. Train NNUE on those scores
4. Repeat (reinforcement learning loop)
5. Slower but you own everything end-to-end

**Recommendation:** Start with Option A, refine with Option B.

### Step 2: PyTorch Training Code

```python
# Pseudocode — write this yourself
class ChessNNUE(nn.Module):
    def __init__(self):
        self.accumulator = nn.Linear(768, 256)  # per perspective
        self.l1 = nn.Linear(512, 32)            # concat both perspectives
        self.l2 = nn.Linear(32, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, white_features, black_features):
        w_acc = clipped_relu(self.accumulator(white_features))
        b_acc = clipped_relu(self.accumulator(black_features))
        x = torch.cat([w_acc, b_acc], dim=1)
        x = clipped_relu(self.l1(x))
        x = clipped_relu(self.l2(x))
        return self.l3(x)  # centipawn-scaled output
```

### Step 3: Training Hyperparameters
- Batch size: 16,384
- Learning rate: 1e-3 with cosine decay
- Loss: MSE on evaluation score, or cross-entropy on game outcome
- Quantization-aware training (QAT) from epoch 50+
- Export int8-quantized weights as binary blob for FPGA

### Dataset Size vs Strength

| Dataset Size | Training Time (T4 GPU) | Expected Strength |
|-------------|----------------------|-------------------|
| 10M positions | ~2-4 hours | Weak (~1800 ELO eval) |
| 50M positions | ~10-20 hours | Moderate (~2200 ELO eval) |
| 200M positions | ~40-80 hours | Strong (~2600 ELO eval) |
| 500M+ positions | ~100-200 hours | Very strong (~2800+ ELO eval) |

### Step 4: Testing the NNUE
1. **Loss metrics** — track validation loss during training
2. **Play against known engines** — use cutechess-cli, 1000+ games at fixed time controls
3. **Tactical test suites** — WAC, ECM, STS (300 puzzles, measure % solved)
4. **Self-play improvement** — each new net should beat previous >55%

---

## 8. Azure Budget Strategy

### Budget: $150/month × 3 months = ~$450 total

All Azure credits are reserved for the large DDR4 model (HalfKP→1024). The BRAM-only net trains locally on the A2000 GPU at $0 cost.

### Local Hardware: NVIDIA A2000 4GB
| Metric | Value |
|--------|-------|
| VRAM | 4 GB GDDR6 |
| CUDA cores | 3,328 |
| Role | BRAM net training, pipeline development, distillation |
| Cost | $0 (already owned) |

### Azure VM Options

| VM Option | $/hr | Hours on $150 | Training Runs |
|-----------|------|---------------|---------------|
| **T4 spot** | ~$0.15 | **~1,000 hrs** | 10-20 full runs |
| T4 on-demand | $0.53 | ~280 hrs | 3-5 full runs |
| A10 spot | ~$0.59 | ~250 hrs | 3-5 runs (faster) |

**Use T4 spot instances.** Save checkpoints every 5 epochs so preemptions don't lose progress.

### Services to Use
- **NC-series / T4 VMs** for GPU training
- **Standard_D series** CPU VMs for bulk data generation (if needed)
- **Azure Blob Storage** for datasets (cheap)

### Workflow
```
Local A2000                          Azure T4 (spot)
───────────                          ─────────────────
1. Build data pipeline
2. Experiment with architectures
3. Train BRAM net (768→256)
4. Test & iterate (fast cycle)
5. Finalize feature encoding
   ──── learned what works ────►
                                     6. Train big net (40K→1024)
                                     7. Scale to 1B+ positions
                                     8. QAT for int8 export
                                     9. Final fine-tuning runs
```

**See [Section 14](#14-revised-nnue-training-plan-10-12-weeks-450-azure) for the detailed run-by-run spending plan.**

---

## 9. Testing & ELO Estimation

### ELO Breakdown

| Factor | Desktop Stockfish | Kria Engine |
|--------|-------------------|-------------|
| NNUE quality | Full HalfKAv2 | Your architecture |
| Search speed | ~80-100M nps | ~1-5M nps (ARM + FPGA) |
| Base ELO | ~3600 | — |

### Realistic Projections

**BRAM-only net:**
| Component | Optimistic | Conservative |
|-----------|-----------|--------------|
| NNUE eval quality | 2700 | 2400 |
| Search (your code, 10 weeks) | -200 | -400 |
| FPGA speed bonus | +100 | +50 |
| Syzygy endgame | +50 | +30 |
| **Estimated ELO** | **~2650** | **~2080** |

**DDR4 accumulator net:**
- Add ~200-400 ELO from stronger eval
- Subtract ~100-200 from slower eval speed
- Net gain: ~100-200 ELO over BRAM-only
- **Estimated: 2500–2900 ELO**

### Biggest Bottleneck
ARM A53 at 1.3GHz for search — ~10-20× slower than desktop i7 for branch-heavy code.

### How to Push Higher
1. Optimize move gen — ARM NEON SIMD
2. Aggressive pruning — LMR, null move, futility, SEE
3. Use R5F cores for TT probing + time management (see [Section 19](#19-cortex-r5f-real-time-cores--architecture--usage))
4. FPGA pipeline — target 1 eval per clock cycle sustained
5. Lazy SMP across 4 ARM cores

---

## 10. Connect Four Engine — Existing Codebase & Chess Porting Guide

> **Original 10-week development pipeline has been superseded by [Section 14](#14-revised-nnue-training-plan-10-12-weeks-450-azure) (training-focused) and [Section 15](#15-post-training-work-after-internship) (post-training integration).**

### Source Code: [github.com/Isaac-Ernst/Connect-Four-AI](https://github.com/Isaac-Ernst/Connect-Four-AI)

### What You Already Have (Transfers to Chess ~60-70%)
| Component | Connect Four Implementation | Chess Port Status |
|-----------|---------------------------|-------------------|
| Negamax + Alpha-Beta | ✅ Full implementation | Direct port — same algorithm |
| MTD(f) | ✅ With PVS fallback | Direct port — works identically |
| Principal Variation Search (PVS) | ✅ Used in hybrid mode | Direct port |
| Late Move Reductions (LMR) | ✅ Reduces non-promising moves | Direct port — tune reduction table for chess |
| Lockless Transposition Table | ✅ Packed 64-bit entries (sig\|score\|depth\|move\|flag) | Port with changes: need Zobrist hashing, larger table (64MB+) |
| History Heuristic | ✅ Move ordering by success rate | Direct port — same concept |
| Iterative Deepening | ✅ Implicit in MTD(f) loop | Direct port |
| Bitboard Representation | ✅ 64-bit, column-major with ghost row | Redesign: 12 bitboards (one per piece type), magic bitboards for sliding pieces |
| Board Symmetry | ✅ Mirror reduction for left/right | Limited in chess: only some positions have symmetry |
| Multi-threaded Book Gen | ✅ 7 threads, DFS | Adapt for Lazy SMP search threads |

### What's New for Chess (Must Build From Scratch)
| Component | Why It's Needed | Difficulty |
|-----------|----------------|------------|
| **Move Generation** | Magic bitboards for sliding pieces, pawn rules, castling, en passant | High — most code |
| **Zobrist Hashing** | Random number XOR for incremental hash — needed for TT and opening book | Medium |
| **Quiescence Search** | Search captures/checks at leaf nodes to avoid horizon effect | Medium — critical for quality |
| **Null Move Pruning** | Skip a turn, search at reduced depth — huge pruning gains | Medium |
| **Static Exchange Evaluation (SEE)** | Determine if a capture sequence is winning — for move ordering | Medium |
| **UCI Protocol** | Universal Chess Interface — required for cutechess-cli testing | Medium — text protocol, ~500 lines |
| **NNUE Integration** | Replace material eval with FPGA/NNUE call | Medium |
| **Lazy SMP** | Multi-threaded search across 4 ARM cores (different from book gen threads) | Medium |

### UCI Protocol — Required for Testing
Your engine **must** implement the [Universal Chess Interface (UCI)](https://www.chessprogramming.org/UCI) protocol to work with cutechess-cli for automated tournament testing. UCI is a simple text-based protocol:
```
GUI → Engine:  "uci"           → Engine responds with name, options
GUI → Engine:  "isready"       → Engine responds "readyok"  
GUI → Engine:  "position ..."  → Set up position (FEN or moves)
GUI → Engine:  "go depth 20"   → Start searching
Engine → GUI:  "info depth 15 score cp 45 pv e2e4 e7e5 ..."  → Search info
Engine → GUI:  "bestmove e2e4" → Final answer
```
Implement UCI early — it's the interface to your entire testing infrastructure.

### Incremental ELO Testing — Version Progression Table

Test each version against the previous in a 1000-game tournament (cutechess-cli, 10s+0.1s time control). Use SPRT for early stopping when the result is conclusive.

| Version | What Changed | Expected ELO | vs Previous |
|---------|-------------|-------------|-------------|
| v0 | Material + PST eval only, alpha-beta | ~1200 | — baseline |
| v1 | + Iterative deepening + aspiration windows | ~1400 | +200 |
| v2 | + Null move pruning + LMR | ~1600 | +200 |
| v3 | + Quiescence search + SEE move ordering | ~1800 | +200 |
| v4 | + Transposition table (Zobrist, 64MB) | ~1900 | +100 |
| v5 | + BRAM NNUE eval (replace material+PST) | ~2300 | +400 |
| v6 | + Lazy SMP (4 threads) | ~2450 | +150 |
| v7 | + DDR4 NNUE eval (large net) | ~2700 | +250 |
| v8 | + Syzygy endgame tables | ~2750 | +50 |
| v9 | + Opening book + fine-tuning | ~2800 | +50 |

> **Reality check on these numbers.** These are optimistic textbook estimates. Real engines rarely stack improvements this cleanly:
> - v4 (TT) typically adds **+150-250 ELO**, not +100 — TT helps enormously with iterative deepening efficiency. Underestimated here.
> - v5 (NNUE) adding +400 assumes both that your search is eval-limited AND that the BRAM net is well-trained. Realistic range: **+250-450**.
> - v6 (Lazy SMP) on ARM A53 often yields **+80-120**, not +150 — the A53's smaller L2 and shared memory bandwidth cap thread-scaling at ~1.7-2.5×, not 4×.
> - The cumulative path v0 → v9 requires every increment to land. Real from-scratch engines typically see **60-75% of textbook gains**, especially on ARM-class CPUs.
>
> **Realistic ceiling on KV260 hardware: ~2500-2700 ELO** (BRAM net + good search) or **~2700-2850 ELO** (DDR4 net). Treat 2900+ as a stretch goal, not a baseline.

**Testing tools:**
- [cutechess-cli](https://github.com/cutechess/cutechess) — automated engine-vs-engine matches
- SPRT (Sequential Probability Ratio Test) — early stopping: `cutechess-cli -sprt elo0=0 elo1=10 alpha=0.05 beta=0.05`
- Reference opponents: Stockfish at reduced depth/hash, Ethereal, Laser, or self-play

---

## 11. Parts List & Purchase Links

### Required Parts

| # | Part | Price | Link |
|---|------|-------|------|
| 1 | AMD Kria KV260 Vision AI Starter Kit | ~$280 | [DigiKey](https://www.digikey.com/en/products/detail/amd/SK-KV260-G/13985269) |
| 2 | KV260 Power Supply (HW-PSA01-SK-G, 12V/3A) | ~$33 | [DigiKey](https://www.digikey.com/en/products/detail/amd/HW-PSA01-SK-G/14280969) |
| 3 | 7" Touchscreen (SunFounder TS-7 Pro, 1024×600) | ~$60 | [SunFounder](https://www.sunfounder.com/products/ts-7-pro-7-inch-touch-screen) |
| 4 | 256GB NVMe SSD (for Syzygy tablebases) | ~$35 | [Newegg](https://www.newegg.com/p/pl?d=256GB+nvme+SSD) |
| 5 | NVMe USB 3.1 Enclosure | ~$20 | [UGREEN](https://us.ugreen.com/collections/ssd-enclosures-fast-secure-storage) |
| 6 | 32GB MicroSD (SanDisk Ultra, Class 10) | ~$8 | [Amazon](https://www.amazon.com/micro-sd-card-32gb-class-10/s?k=micro+sd+card+32gb+class+10) |
| 7 | 3D-printed case | Free | Microsoft 3D printers |
| 8 | Azure GPU training (T4 spot) | $0 | $450 credits ($150/month × 3) |
| | **TOTAL** | **~$436** | |

### Optional Upgrades

| Part | Price | Why |
|------|-------|-----|
| USB keyboard | ~$10 | Debug on-device |
| HDMI + USB-C cables | ~$10 | Screen connection |
| 40mm 12V fan | ~$5 | Cooling for sustained FPGA load |

### Alternative Touchscreen Options
- Eyoyo 7" (1024×600, HDMI+USB): [Amazon](https://www.amazon.com/Eyoyo-Raspbery-Screen-7-inch-Touchscreen/dp/B0F9NWWG9L)
- DFRobot 7" HDMI: [DFRobot](https://www.dfrobot.com/product-1655.html)

### Syzygy Endgame Tablebase Downloads

6-piece tablebases: ~68 GB (WDL) + ~81 GB (DTZ) = **~149 GB total** → fits on the 256GB SSD.

| Source | Link | Notes |
|--------|------|-------|
| HTTP (Sesse.net) | http://tablebase.sesse.net/syzygy/6-men/ | Direct download, use wget for bulk |
| HTTP (Lichess mirror) | https://tablebase.lichess.ovh/tables/standard/6-wdl/ | Alternative mirror |
| Torrent — WDL | https://archive.org/details/Syzygy6MenWDL | Internet Archive, reliable |
| Torrent — DTZ | https://archive.org/details/Syzygy6MenDTZ | Internet Archive, reliable |
| Downloader tool | https://github.com/jj-jaguar/Syzygy-Tablebase-Downloader | Automated, resumes, multi-mirror |

**Tips:**
- Download WDL (win/draw/loss) first — it's used during search for pruning
- DTZ (distance-to-zero) is used for actual move selection in endgames
- Verify checksums after download
- Store on SSD, accessed via USB 3.0 — latency is fine since Syzygy is only probed at ≤6 pieces

### Training Data Downloads

| Source | Link | Content | Size |
|--------|------|---------|------|
| **Lichess Open Database** | https://database.lichess.org | All rated games by month, PGN.zst format | ~20-50 GB/month compressed |
| Lichess file list | https://database.lichess.org/standard/list.txt | Index of all available monthly files | — |
| Example: April 2026 | https://database.lichess.org/standard/lichess_db_standard_rated_2026-04.pgn.zst | Single month | — |
| **CCRL Archives** (40/15) | http://www.computerchess.org.uk/ccrl/4040 | Computer chess games, strong play | Varies |
| CCRL Blitz (2m+1s) | http://www.computerchess.org.uk/ccrl/404 | Faster time control games | Varies |
| **Leela Chess Zero** | https://lczero.org | Self-play training data | Large |

**Recommended approach for training data:**
1. Download 3-6 months of Lichess data (games rated 2000+)
2. Decompress with `zstd -d filename.pgn.zst`
3. Filter with your PGN parser: keep games where both players are 2000+ rated
4. Extract positions + game outcomes → binary training format
5. Target: 200M-1B positions for large model training

### Opening Book

#### What Is It?
A precomputed database of strong opening moves so the engine plays book theory instantly (no search needed) for the first 10-20 moves. Stored in **Polyglot format** — a simple binary format using Zobrist hash → (move, weight, learn).

#### Where It Lives on Hardware
```
32GB MicroSD Card:
├── PetaLinux boot image
├── Chess engine binary
├── NNUE weight files (int8 blobs)
├── opening_book.bin          ← HERE (~5-50 MB)
└── Config files, logs
```

The opening book is tiny (5-50 MB) and accessed only at game start, so it lives on the **MicroSD** alongside the engine binary. No need to put it on the SSD — that's reserved for the 149 GB of Syzygy tables.

#### How to Create Your Opening Book

**Option 1: From Lichess master games (recommended)**
```bash
# 1. Download Lichess elite database (games from 2400+ rated players)
#    https://database.lichess.org
#    Filter for games where both players are 2400+

# 2. Use the Polyglot tool to generate the book
git clone https://github.com/lichess-org/polyglot
cd polyglot && make

# 3. Convert PGN → Polyglot binary
polyglot make-book \
    -pgn master_games.pgn \
    -bin opening_book.bin \
    -min-game 10 \        # Move must appear in ≥10 games
    -max-depth 30         # Up to 30 half-moves (15 full moves)
```
- Polyglot tool: https://github.com/lichess-org/polyglot

**Option 2: Write your own book generator (since you're doing everything from scratch)**
The Polyglot format is simple:
```
Each entry = 16 bytes:
  - key:    8 bytes (Zobrist hash of position)
  - move:   2 bytes (encoded from/to/promotion)
  - weight: 2 bytes (how often this move was played)
  - learn:  4 bytes (optional learning data)

File = sorted array of entries, binary searchable by key
```
You can write your own PGN → Polyglot converter in a few hundred lines of code.
This is a good exercise and fits your "everything from scratch" philosophy.

**Option 3: Build from self-play (post-training)**
Once your NNUE is trained, run your engine on common openings at high depth.
Record the best moves found → add to book. This creates a book tuned to your engine's strengths.

#### Probing the Book at Runtime
```
1. Compute Zobrist hash of current position
2. Binary search the .bin file for matching key
3. If found: select move weighted by frequency (add some randomness for variety)
4. If not found: exit book, start normal search
```

Book probing is essentially free — one binary search on a small file loaded in memory at boot.

---

## 12. Admissions Strategy

### What Stanford/MIT/CMU/Harvard Actually Care About

Top graduate programs evaluate applicants on **research potential first, technical depth second**. They see hundreds of competent engineers; what stands out is evidence you can frame a question, design an experiment, and produce a defensible finding. Lead with the research framing.

### The Research Contribution (Lead Here)

The strongest framing of this project:

> *"An empirical study of NNUE evaluation quality vs throughput on resource-constrained FPGA hardware, comparing five neural-network architectures with different parameter counts, memory hierarchies, and training procedures, all deployed on a single $280 development board."*

That's a thesis statement. It has a hypothesis (architecture/memory tradeoffs matter), a methodology (5-way comparison with a Stockfish-distilled control), a constraint (fixed hardware), and a deliverable (measured results). Frame everything else as supporting evidence.

Subordinate to this, you also demonstrate:
- **Hardware-aware ML co-design**: quantization, layer sizes, and feature encodings chosen to fit specific FPGA resources
- **End-to-end systems engineering**: ML training → custom HDL inference engine → embedded Linux runtime → UI
- **Reproducible methodology**: five comparable models with identical pipelines, including a Stockfish-distilled baseline for calibration

### The "From Scratch" Story (Supporting Evidence, Not the Headline)

"I wrote every line of code, no Stockfish derivatives" is supporting evidence of depth, not the central pitch. An admissions reviewer who knows ML will note that the strongest small NNUEs in practice ARE Stockfish-distilled — positioning "from scratch" as the goal can sound naive. Reframe it as scientific rigor:

> *"I implemented an independent training pipeline AND included a Stockfish-distilled baseline as a control. This lets me quantify how much of the small-net ELO comes from the architecture itself vs from teacher quality — a question the existing literature handles informally."*

That sounds like a researcher, not a hobbyist.

### Quantifiable Results (Required)

Every claim must come with a number and a confidence interval:
- "NNUE-on-FPGA achieves X million evals/sec at Y watts (Z× perf/watt vs ARM A53 baseline)"
- "Independent BRAM net achieved A% (±confidence interval) of Stockfish-distilled performance with B× less training compute"
- "Search engine reaches depth N at time control T, with M% pruning rate from null move + LMR + SEE"
- "Total system: $436 BOM, 12W power, ELO E ± confidence interval over 1000-game tournament"

### Technical Domains Covered

| Domain | Evidence |
|--------|----------|
| **Machine Learning** | Independent NNUE training pipeline, QAT, knowledge distillation, 5-model comparison |
| **Computer Architecture** | FPGA inference design (see §20), DSP/BRAM/UltraRAM allocation, AXI interface |
| **Systems Programming** | Chess engine in C++, ARM optimization, bitboards, multi-threaded Lazy SMP |
| **Hardware Design** | SystemVerilog + HLS, timing closure at 200 MHz, Vivado workflow |
| **Cloud Computing** | Azure GPU training pipeline, spot-instance management, blob-storage workflow |
| **Embedded Systems** | PetaLinux, OpenAMP, R5F real-time cores, AXI/DMA |
| **HCI** | Touchscreen UI design |

### Deliverables for Application Portfolio

1. **GitHub monorepo** with clean subprojects:
   - `engine/` — C++ chess engine
   - `nnue-trainer/` — Python training pipeline (PyTorch)
   - `fpga/` — SystemVerilog + HLS + Vivado project
   - `runtime/` — Linux driver + glue code
   - `ui/` — touchscreen application
   - `paper/` — LaTeX source
   - Commit history showing iteration, not a single "initial commit"

2. **Technical paper** (6-8 pages, IEEE/ACM format)
   - Submit to an FPGA-focused workshop (FCCM, FPL, FPT) or post to ArXiv
   - Cite by URL in application materials — gives reviewers a concrete artifact to evaluate

3. **Demo video** (60-90 seconds)
   - Device beating a known opponent at known time control
   - One shot of the FPGA pipeline / Vivado synthesis report
   - End with the comparison chart from the paper

4. **Physical artifact**
   - Bring to interviews (especially on-site for Stanford/MIT/CMU)
   - Tangible objects are remembered far longer than slides

### School-Specific Emphasis

| School | Angle to Emphasize |
|--------|-------------------|
| **Stanford** | ML/systems co-design. Faculty in SAIL + EE (e.g., Kunle Olukotun) value hardware-aware ML. The empirical comparison framing fits their style. |
| **MIT** | First-principles hardware design. CSAIL hardware groups (e.g., Joel Emer's area) value the FPGA work specifically and rigorous benchmarking. |
| **CMU** | Computer architecture + systems performance engineering. Lean hard on throughput/latency/perf-per-watt numbers. |
| **Harvard** | Interdisciplinary breadth. The "I bridged ML, systems, hardware, and HCI in a single artifact" pitch lands well. |

### What NOT To Claim

Avoid these in application materials — they signal inexperience:
- ❌ "Stockfish-level" — you're not, and reviewers know
- ❌ "State-of-the-art" — you're not aiming for SoTA, you're producing a defensible finding
- ❌ "Optimal" — say "tuned to the constraints" instead
- ❌ Excessive precision on ELO numbers without confidence intervals — always report ±CI from SPRT runs

---

## 13. Key References & Resources

### Papers
- *"Efficiently Updatable Neural-Network-based Evaluation Functions"* — Yu Nasu (2018), original NNUE paper
- *"FPGA Implementation of Neural Networks"* — various IEEE papers on quantized inference
- *"Deep Learning on FPGAs: Past, Present, Future"* — survey paper

### Code/Repos to Study (for concepts, not to copy)
- `official-stockfish/nnue-pytorch` — NNUE training reference
- `glinscott/nnue-pytorch` — older fork with good documentation
- `jw1912/bullet` — fast NNUE trainer in Rust
- Xilinx/AMD Vitis AI — FPGA neural network toolchain

### Books
- *Chess Programming Wiki* — https://www.chessprogramming.org (encyclopedic reference)
- *"FPGA Prototyping by SystemVerilog Examples"* — Pong Chu
- *"Neural Networks on FPGAs"* — Jejeesh & Thomas

### Data Sources
- **Lichess database:** https://database.lichess.org (billions of games, open)
- **CCRL game archives:** https://ccrl.chessdom.com
- **Leela Chess Zero data:** https://lczero.org

### HDL/FPGA Tools
- Vivado (AMD/Xilinx) for synthesis
- Cocotb or Verilator for testbenches
- HLS (High-Level Synthesis) — write C++, generate RTL

### Testing Tools
- cutechess-cli — automated engine-vs-engine matches
- Perft — move generation correctness verification
- WAC/ECM/STS — tactical puzzle test suites

---

## 14. Revised NNUE Training Plan (10-12 Weeks, ~$450 Azure)

### Budget
- $150/month × 3 months = ~$450 total
- T4 spot at ~$0.15/hr = **~3,000 GPU-hours**
- Local A2000 4GB for development and small runs (unlimited, $0)

### Strategy: Train Small First, Then Large

**Key insight: Train the BRAM net (small) first, then the DDR4 net (large) separately.**

- **Phase A:** Train BRAM net locally on A2000 — this is your debugging sandbox, learning tool, and safety net
- **Phase B:** Train large net on Azure — informed by everything you learned from Phase A
- **Phase C (optional):** Distill large → small to create an upgraded BRAM net

**Why small first:**
- Debug your entire pipeline at $0 cost (minutes per experiment vs hours on Azure)
- Every bug found locally is money saved on Azure
- The small net is your safety net — even if Azure runs fail, you have a working NNUE
- You learn training fundamentals (loss functions, LR schedules, data quality) on cheap hardware

**Will the small net be weaker than a distilled one?**
- Independent small net: ~2400-2600 eval ELO
- Distilled from large net: ~2500-2700 eval ELO
- Difference: ~50-100 ELO — modest, and distillation can be done later (cheap, ~2-4 hrs on A2000)

**Both models are trained independently with the same data pipeline, then optionally the small net gets a distillation pass at the end. This gives you two fully validated NNUEs.**

### Epoch Budget (Applies to All Runs Below)

The cosine LR schedule needs an explicit `T_max`. Use these defaults:

| Net | Epochs | Notes |
|-----|--------|-------|
| BRAM net (200K params, 200M positions) | **80-120** | float32 for epochs 0-50, then QAT for epochs 50-100 |
| Large net (40M params, 500M-1B positions) | **40-80** | float32 for epochs 0-30, then QAT for the rest. More data → fewer epochs (network sees each position fewer times but more total updates) |
| Distillation (large → BRAM) | **20-40** | Fewer epochs needed — soft targets give stronger signal per step |
| Self-play fine-tuning | **10-20** | Small dataset of high-quality positions, don't overtrain |

> Save checkpoints every 5 epochs to Blob Storage. Run a 5-epoch LR/loss-function comparison BEFORE any full-length run (see Deep Dive Guide §6).

### Phase 1: Infrastructure (Weeks 1-2, Local Only)

**Goal:** Bulletproof training pipeline before spending any Azure credits.

- [ ] PGN parser — handle all notation edge cases
- [ ] Feature encoding — implement both 768 (piece-square) and HalfKP (40,960)
- [ ] PyTorch NNUE — both architectures in same codebase, config-selectable
- [ ] Data loader — efficient binary format, memory-mapped for large datasets
- [ ] Training loop — mixed precision, gradient clipping, cosine LR schedule
- [ ] Checkpoint save/resume — test by killing and restarting
- [ ] QAT pipeline — quantization-aware training → int8 export
- [ ] Validation framework — hold-out set loss tracking, position eval spot-checks
- [ ] Unit tests — feature encoding correctness, forward pass matches manual calc
- [ ] Azure setup script — one command: install deps, mount blob, start training

**Lichess data download:** Start downloading during week 1 (takes time):
- https://database.lichess.org
- Target: all games rated 2000+, last 2-3 years
- Process into binary training format locally

**Validation on A2000:**
- Train BRAM net (768→256) on 10M positions — verify loss decreases
- Train large net (HalfKP→1024) on 1M positions — verify it runs, no OOM
- Checkpoint resume test on both

### Phase 2: BRAM Net — Solid Baseline (Weeks 3-4, Local A2000)

**Goal:** Fully trained small net as your baseline, safety net, and learning tool.

- Train 768 → 256 → 32 → 32 → 1 on 200M positions
- ~8-15 hours per training run on A2000
- Run 3-5 experiments:
  - Run 1: MSE loss on Lichess eval scores
  - Run 2: Cross-entropy loss on game outcomes
  - Run 3: Combined loss (weighted sum)
  - Run 4: Best loss + different LR schedule
  - Run 5: Best config + data augmentation (board flips)
- QAT → export int8 weights
- Basic strength test: plug into a simple alpha-beta search, play vs known engines

**Deliverable:** Validated int8 BRAM net, known baseline strength.

**What you learn here that saves Azure money:**
- Which loss function works best (MSE vs CE vs combined)
- What learning rate schedule converges fastest
- Whether your data pipeline has bugs (feature encoding errors, label issues)
- How many positions are needed before overfitting
- What batch size your training loop handles efficiently

### Phase 3: Minimal Chess Search (Week 4-5, Local)

**Goal:** Build a basic alpha-beta search that can play legal chess with your BRAM NNUE. This is NOT about building a strong engine yet — it's about enabling self-play data generation.

- Basic bitboard move generator (test with perft)
- Alpha-beta with iterative deepening, depth 6-8
- Simple move ordering (captures first, then killer moves)
- Plug in your BRAM NNUE as eval function
- Validate: plays legal chess, doesn't crash, eval scores are reasonable

**Why now:** You need this search to generate self-play data for the large model training. It doesn't need to be fast or fully optimized — just functional.

### Phase 4: Large Model Training (Weeks 5-9, Azure)

**Goal:** Train the strongest possible HalfKP → 1024 net.

**Month 1 Azure ($150 = ~1,000 T4-spot hours):**

| Run | Config | Data | Hours | Purpose |
|-----|--------|------|-------|---------|
| 1 | HalfKP→512 | 200M | 40 | Validate HalfKP encoding at scale |
| 2 | HalfKP→1024 | 500M | 100 | First full-size training |
| 3 | HalfKP→1024 | 500M | 100 | Different LR / loss function (use what worked on BRAM net) |
| 4 | Best of 2-3, resume | +200M new data | 50 | Continue training best checkpoint |
| — | Overhead / debugging | — | 30 | Buffer |
| **Total** | | | **320** | **~$48 spent** |

Save remaining ~$100 for month 2 — you're being methodical, not rushed.

**Month 2 Azure ($150 + ~$100 carried = ~$250 = ~1,600 hours):**

| Run | Config | Data | Hours | Purpose |
|-----|--------|------|-------|---------|
| 5 | Best arch | 1B positions | 200 | Scale up data |
| 6 | HalfKAv2 features | 500M | 100 | Test king-bucket features |
| 7 | Best of 5-6, fine-tune | 200M curated | 50 | High-quality positions only (2400+ games) |
| 8 | Self-play data gen (your engine from Phase 3) | — | Local | Generate 50M+ positions from best net |
| 9 | Best + self-play data | 500M mixed | 100 | Reinforcement refinement |
| **Total** | | | **~450** | **~$68 spent** |

**Month 3 Azure ($150 + ~$180 carried = ~$330 = ~2,200 hours):**

| Run | Config | Data | Hours | Purpose |
|-----|--------|------|-------|---------|
| 10 | Second self-play cycle | 500M mixed | 100 | Stronger self-play data |
| 11 | Architecture variations | various | 150 | Final architecture search |
| 12 | Final training — large net | All best data | 200 | Definitive large model |
| 13 | QAT final passes — large net | — | 20 | int8 export for DDR4 |
| 14 | Knowledge distillation → BRAM net | — | 30 | Large net → upgraded BRAM net |
| 15 | QAT final pass — distilled BRAM net | — | 5 | int8 export for BRAM |
| **Total** | | | **~505** | **~$76 spent** |

### Phase 5: Validation & Export (Weeks 10-12)

- **Tournament testing:** Run best nets against each other and reference engines
  - Use your Phase 3 search engine with cutechess-cli
  - 1000+ games per matchup at fast time controls
  - Test both BRAM net (independent) vs BRAM net (distilled) — measure the ~50-100 ELO difference
- **Tactical accuracy:** Test on WAC/ECM/STS suites
  - Measure: does the net correctly identify the best move in tactical positions?
- **Export final weights:**
  - Large net: int8 quantized binary blob for DDR4 loading
  - BRAM net (independent): int8 blob — your safety net
  - BRAM net (distilled): int8 blob — your best small net
  - All three verified against PyTorch float32 outputs (max error < 2 centipawns)
- **Archive everything:**
  - All checkpoints from all 15+ runs
  - Training logs and loss curves
  - Configs that produced each net
  - The training code (frozen commit hash)
  - Processed datasets in binary format
  - Self-play game archives
  - Store on local drive + external backup

### Knowledge Distillation Process (Large → Small, Phase 4 Run 14)

```
1. Take trained large net (HalfKP → 1024) — the "teacher"
2. Generate evaluations for 50M+ positions using the teacher
3. Train BRAM net (768 → 256) to match the teacher's outputs
   - Loss = MSE(small_net(pos), large_net(pos))
   - The small net learns compressed knowledge from the large net
4. Fine-tune with a blend: 70% teacher labels + 30% game outcomes
5. QAT → export int8
```

This typically gives 50-100 ELO improvement over training the small net independently.
Can be done locally on A2000 (~2-4 hours) or on Azure if still in budget.

### Experiment Tracking

Use a simple spreadsheet or Weights & Biases (free tier):

| Run ID | Architecture | Features | Data Size | LR | Loss | Best Val Loss | ELO Estimate | Notes |
|--------|-------------|----------|-----------|-----|------|--------------|-------------|-------|
| 001 | 768→256 | PS | 200M | 1e-3 | MSE | 0.0234 | ~2400 | baseline |
| 002 | HalfKP→1024 | HalfKP | 500M | 1e-3 | MSE | 0.0189 | ~2700 | first large |
| ... | | | | | | | | |

### Critical Rules for Azure Spending

1. **Never start a run without testing it locally first** (even 1 epoch on 1M positions)
2. **Always use spot instances** — set up auto-resume from checkpoints
3. **Save checkpoints to Blob Storage every 5 epochs** — preemptions lose unsaved work
4. **Monitor costs daily** — set Azure budget alerts at $40, $80, $120 per month
5. **Don't over-optimize early** — first priority is getting a working large net, then iterate
6. **Download everything before each month's credits expire**

### Expected Outcomes

| # | Model | Training Source | Teacher | Expected Eval ELO | Size | Target Hardware |
|---|-------|---------------|---------|-------------------|------|----------------|
| 1 | Handcrafted eval (material + PST) | Your code | None | ~1200-1500 | N/A | CPU |
| 2 | BRAM NNUE (independent) | Lichess W/L/D | None — 100% yours | 2400-2600 | ~413 KB | FPGA BRAM |
| 3 | DDR4 NNUE (large) | Lichess W/L/D + self-play | None — 100% yours | 2800-3000 | ~40 MB | FPGA + DDR4 |
| 4 | BRAM NNUE (distilled from YOUR large net) | Model 3's evaluations | Model 3 — yours | 2500-2700 | ~413 KB | FPGA BRAM |
| 5 | BRAM NNUE (distilled from Stockfish) | Stockfish depth 8-10 evals | Stockfish — baseline | 2550-2750 | ~413 KB | FPGA BRAM |

**Model 5 exists purely as a comparison baseline.** The gap between Model 4 and Model 5 measures how close your independently trained large net gets to Stockfish's evaluation quality when compressed to the same architecture. This is a publishable finding.

### Stockfish Pre-Evaluated Training Data

You do NOT need to run Stockfish yourself for Model 5. Two sources of pre-labeled data exist:

| Source | Link | Format | Notes |
|--------|------|--------|-------|
| **Stockfish official training data** | https://github.com/official-stockfish/data | `.binpack` binary | Positions + Stockfish evals used to train official nets. Need `binpack` parser to extract FEN+eval pairs. |
| **Lichess evaluated positions** | https://database.lichess.org/#evaluation | JSON | Server-side Stockfish analysis on millions of games. FEN + eval + depth + PVs. Directly usable with Python. |
| Lichess Game Export API | `https://lichess.org/game/export/{id}?evals=true` | JSON/PGN | Per-game evals for analyzed games. Rate-limited — not for bulk, but useful for spot-checking. |

**Recommended for Model 5:** Use the Lichess evaluated positions database (JSON format). Download the evaluation dump, filter for positions analyzed at depth ≥ 12, extract FEN + centipawn score, and train your BRAM architecture on those scores using MSE loss. This avoids running Stockfish yourself entirely.

**Alternative (generate yourself):** Run Stockfish at depth 8 on 50M positions from your existing Lichess PGN data. At depth 8: ~0.02s/position × 50M = ~12 days on 8 threads. Run in background on desktop during Azure training weeks.

### Summer Deliverables Checklist

```
Training Outputs (Weeks 1-12):
  ✅ Model 1: Handcrafted eval function (material + PST, written in code)
  ✅ Model 2: BRAM NNUE — independent, trained on Lichess W/L/D (int8, ~413 KB)
  ✅ Model 3: DDR4 NNUE — large, trained on Lichess + self-play (int8, ~40 MB)
  ✅ Model 4: BRAM NNUE — distilled from Model 3 (int8, ~413 KB)
  ✅ Model 5: BRAM NNUE — distilled from Stockfish evals (int8, ~413 KB, baseline)
  ✅ Minimal self-play engine (bitboard + alpha-beta + NNUE eval, for data gen)
  ✅ Complete training pipeline (PGN parser, feature encoder, PyTorch code)
  ✅ All checkpoints, training logs, loss curves, configs archived locally

Post-Summer (no budget constraint):
  ❌ Full chess engine with all search optimizations
  ❌ UCI protocol for cutechess-cli tournament testing
  ❌ FPGA inference design (SystemVerilog/VHDL)
  ❌ Integration (search + FPGA eval + Syzygy + opening book)
  ❌ Touchscreen UI
  ❌ 3D-printed case (print at Microsoft before leaving!)
  ❌ ELO tournament testing all 5 eval functions
  ❌ Technical paper + demo video

Post-Summer Testing Matrix:
  Build ONE search engine, swap eval functions, test all 6 configs:
    Config A: Handcrafted (material + PST)          → baseline
    Config B: BRAM NNUE (independent)                → your small net
    Config C: DDR4 NNUE (large)                      → your large net
    Config D: BRAM NNUE (distilled from yours)       → compressed from your large net
    Config E: BRAM NNUE (distilled from Stockfish)   → baseline comparison
    Config F: Stockfish itself                       → reference
  
  Each config tested on:
    - Desktop (i7 + 32GB) → pure software comparison
    - KV260 (ARM A53 + FPGA) → embedded hardware comparison
  
  1000-game tournaments for each matchup via cutechess-cli
```

---

## 15. Post-Training Work (After Internship)

After the 10-12 week training focus, you'll have both NNUE models validated and exported. The remaining work splits into two roughly parallel tracks: software (chess engine completion) and hardware (FPGA implementation). **Be honest with yourself: this is another 6-12 months of part-time work, not a few weekends.**

### Time Budget Reality Check

Assuming ~15 hrs/week of post-internship time:

| Track | Subtasks | Estimated Hours | Calendar Time |
|-------|----------|----------------|---------------|
| Software: chess engine | Move gen + full search + UCI + tuning | 150-250 hrs | 10-17 weeks |
| Hardware: FPGA NNUE | See §20.10 — RTL, HLS, verification, timing closure | 200-350 hrs | 13-23 weeks |
| Hardware: touchscreen UI | Qt or LVGL + game logic + visuals | 40-80 hrs | 3-5 weeks |
| Hardware: 3D case | CAD + iterative prints | 20-40 hrs | **MUST happen during internship** |
| Validation | 2000+ games × multiple configs + bug fixes | 60-100 hrs | 4-7 weeks |
| Paper + demo + docs | Writing, figures, video, README cleanup | 60-100 hrs | 4-7 weeks |
| **Total** | | **530-920 hrs** | **6-12 months at 15 hrs/week** |

The software and FPGA tracks can run in parallel because they only need integration in month 5-7.

### Detailed Breakdown

**1. Complete chess engine** (~150-250 hrs)
- Magic bitboards for sliding pieces: ~30 hrs
- Full pseudo-legal + legal move generator + perft validation: ~30 hrs
- Iterative deepening + aspiration windows: ~10 hrs
- Quiescence search + SEE: ~20 hrs
- Null move + LMR + futility pruning: ~30 hrs
- Multi-threaded Lazy SMP across 4 ARM cores: ~25 hrs
- Move ordering (TT move, killers, history, captures): ~15 hrs
- Time management: ~10 hrs
- Tuning + regression testing: 20-80 hrs

**2. UCI protocol** (~15-25 hrs)
- Implement the surface used by cutechess-cli, Arena, Banksia
- Standard `go`/`stop`/`position`/`setoption` handling
- `info` reporting (depth, score, pv, nps)

**3. FPGA inference design** (~200-350 hrs)
- See [§20](#20-fpga-inference-pipeline--architecture--implementation) for the detailed plan
- This is the highest-risk, highest-novelty work — **start it early, don't leave to last**

**4. System integration** (~30-50 hrs)
- Linux driver (mmap'd AXI region or character device)
- C++ wrapper class for FPGA NNUE
- Replace software NNUE eval with FPGA call
- Syzygy probe integration (libsyzygy or minimal probe)
- Opening book lookup at game start

**5. Touchscreen UI** (~40-80 hrs)
- Qt (heavier, prettier) or LVGL (lighter, embedded-friendly)
- Board rendering, drag/drop, move highlighting
- Game clock, move list, eval bar
- Settings UI (depth, time control, NNUE choice)

**6. 3D-printed case** (~20-40 hrs) — **TIME-SENSITIVE**
- Design in Fusion 360 or OnShape
- Include: KV260 mount, touchscreen frame, USB/HDMI/power cutouts, ventilation, fan mount
- Print 2-3 iterations to fix tolerance issues
- **Must complete during internship** — Microsoft printer access ends with badge access
- **Start case design in Week 2-3 of internship, not Week 11**

**7. ELO testing** (~60-100 hrs)
- Set up cutechess-cli with all 5+ NNUE configs
- Tournament time control (e.g., 60+0.6s)
- 2000+ games per matchup for statistical significance
- Run against reference engines: weak Stockfish (depth-limited), Ethereal, Laser, or earlier self
- SPRT for early stopping

**8. Documentation + paper + demo** (~60-100 hrs)
- GitHub repo cleanup, README, architecture docs
- 6-8 page technical paper (LaTeX/IEEE format)
- Submit to a relevant workshop or ArXiv
- 1-2 minute demo video showing device playing
- Project website / portfolio page

### Recommended Schedule (9 Months Post-Internship)

```
Month 1-2:  Chess engine: move gen, basic search, UCI, perft validation
Month 1-3:  FPGA: PyTorch bit-accurate reference + first RTL modules
Month 3-5:  Chess engine: full pruning suite, multi-threaded search
Month 3-6:  FPGA: full pipeline, AXI integration, Linux runtime
Month 5-7:  Touchscreen UI + system integration on KV260
Month 7-8:  ELO testing (2000+ games per matchup) + bug fixes
Month 8-9:  Paper writing, demo video, portfolio cleanup
```

### Time-Sensitive Checklist (Before Internship Ends)

| Item | Deadline | Why |
|------|----------|-----|
| 3D-printed case | Last 2 weeks of internship | Microsoft printer access ends with badge |
| Azure artifact migration | Last week | Move all trained checkpoints + datasets off Azure storage to local + external backup |
| Azure VM teardown | Last day | Avoid charges to your personal account when intern credits expire |
| Local backup of EVERYTHING | Last week | Code, weights, training data, configs, logs, intermediate artifacts |
| Final Lichess data download | Last 2 weeks | Lichess data is open but downloading 100+ GB takes time |

---

## 16. Decisions Made & Rationale

| # | Decision | Alternatives Considered | Rationale |
|---|----------|------------------------|-----------|
| 1 | **KV260 over KR260** | KR260 ($350) | Same XCK26 FPGA. KR260 adds CAN/TSN for robotics — unnecessary. Save $70. |
| 2 | **NVMe SSD + USB 3.0 enclosure** | SATA SSD, bare NVMe | NVMe is future-proof if upgrading to M.2 carrier board. USB 3.0 fine for Syzygy. Marginal cost vs SATA. |
| 3 | **Train small net first** | Train large first, skip small | Debug pipeline at $0. Small net = safety net. Every bug found locally saves Azure money. |
| 4 | **Transposition table in DDR4 only** | TT on SSD for larger capacity | SSD random read latency (~1ms) would reduce engine to ~500 nps. DDR4 at 128MB with good replacement >> 10GB SSD TT. |
| 5 | **USB 3.0 SSD for Syzygy** | Direct M.2/SATA | Syzygy probes are infrequent (~100s/game), small (~1-10KB). USB 3.0 latency is invisible. |
| 6 | **BRAM-only NNUE: 768→256→32→32→1** | Larger 1536→512, HalfKP-lite 6144→256 | ~413KB fits BRAM with 2.9MB to spare for TT state, pipeline registers, buffers. Good balance of strength vs simplicity. |
| 7 | **DDR4 NNUE: HalfKP 40960→1024** | Smaller HalfKP→512 | Full Stockfish-scale for maximum eval strength. DDR4 incremental reads are only 2-8KB per move. |
| 8 | **T4 spot instances on Azure** | T4 on-demand, A10 spot | Best $/hour for training. Spot preemptions handled by checkpoint resume. |
| 9 | **Everything from scratch** | Use Stockfish/Ethereal code | Admissions story is far stronger. Full ownership of every line of code. |
| 10 | **Knowledge distillation (large→small)** | Train small independently only | +50-100 ELO for ~2-4 hours on A2000. The A/B comparison is also a publishable result. |

---

## 17. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **ARM A53 too slow for competitive search** | Caps realistic ELO at ~2500-2700 (BRAM net) / ~2700-2850 (DDR4 net), even with strong eval. 10-20× slower than desktop i7 for branch-heavy code. | **High and largely unavoidable** | Aggressive pruning (LMR, null move, futility, SEE). Lazy SMP across 4 cores (expect 1.7-2.5× scaling, not 4×, due to L2 contention). Lean heavily on strong NNUE eval to reduce needed depth. Use R5F cores for time management + FPGA coordination (not TT probes — see §19). Accept that 2900+ ELO requires faster hardware (Jetson co-processor) or many additional months of optimization. |
| **Azure spot preemptions lose training progress** | Wasted GPU-hours and money | Medium | Save checkpoints every 5 epochs to Azure Blob Storage. Auto-resume scripts. Keep data on Blob, not local VM. |
| **int8 quantization degrades NNUE quality** | Could lose 50-100 ELO from float32 | Medium | Quantization-aware training (QAT) from epoch 50+. Verify int8 output matches float32 within 2 centipawns. |
| **10-12 weeks isn't enough for strong large net** | Weaker eval than target | Medium | BRAM net is the safety net — always have a working engine. Prioritize training runs methodically (see Section 14). |
| **FPGA timing closure at 200MHz** | Slower eval pipeline, fewer evals/sec | Medium | Start with conservative 100MHz design, optimize later. Pipeline the critical path. Use Vivado timing analysis early. |
| **3D printer access ends with internship** | No physical case for demos | Low | Design case in first weeks, submit prints early with buffer time. |
| **BRAM net too weak for competitive play** | ELO significantly below 2200 | Low | 768→256 should be sufficient for 2400+ eval quality. Scaling option (1536→512) still fits in BRAM+UltraRAM. |
| **Data quality issues in Lichess games** | Noisy labels, weak games pollute training | Medium | Filter for 2000+ rated games only. Remove games with clock flag/timeout results. Validate feature encoding with unit tests. |

---

## 18. Alternative Hardware Considered

These were evaluated and may be worth revisiting as the project matures:

| Platform | Price | Pros | Cons | Best For |
|----------|-------|------|------|----------|
| **Kria KV260** (selected) | ~$280 | FPGA + ARM SoC, best admissions story, large community | ARM A53 is slow for search | This project |
| **Jetson Orin Nano Super** | ~$249 | 6× Cortex-A78 cores (much faster search), 67 TOPS GPU, 8GB LPDDR5 | No FPGA — NNUE runs on GPU, less unique | Higher ELO ceiling if search speed is bottleneck |
| **Dual-board: KV260 + Jetson** | ~$530 | FPGA eval + fast CPU search, heterogeneous computing story | Higher cost, more complex integration | Highest ceiling + best admissions narrative |
| **Xilinx ZCU104** | ~$1,900 | 6× more BRAM (912 blocks), 1,728 DSP slices | Way too expensive for student project | Aspirational only |
| **Lattice CrossLink-NX** | ~$50-150 | Very low power, tiny form factor | Too small for NNUE, limited BRAM | Not suitable |

**Recommendation:** Start with KV260. If search speed is the bottleneck after training (likely), consider adding a Jetson Orin Nano as a compute co-processor — the heterogeneous computing angle strengthens the admissions story significantly.

---

## 19. Cortex-R5F Real-Time Cores — Architecture & Usage

### What Are the R5F Cores?

The Kria KV260's Zynq UltraScale+ SoC includes **two ARM Cortex-R5F** cores in addition to the four Cortex-A53 application cores. These are fundamentally different processors designed for different workloads:

```
┌─────────────────────────────────────────────────────────────┐
│  Zynq UltraScale+ XCK26 SoC                                │
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │ Application          │    │ Real-Time Processing Unit   │ │
│  │ Processing Unit (APU)│    │ (RPU)                       │ │
│  │                      │    │                             │ │
│  │  4× Cortex-A53       │    │  2× Cortex-R5F              │ │
│  │  @ 1.3 GHz           │    │  @ 533 MHz                  │ │
│  │  32KB I/D L1 cache   │    │  32KB I/D L1 cache          │ │
│  │  1MB shared L2       │    │  128KB TCM per core         │ │
│  │  MMU (virtual memory)│    │  MPU (no virtual memory)    │ │
│  │  Runs Linux           │    │  Bare-metal or RTOS         │ │
│  │  Out-of-order          │    │  In-order, deterministic   │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Programmable Logic (FPGA)                             │   │
│  │  256K logic cells, 1,248 DSP, 3.3MB BRAM+UltraRAM    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  All three domains share access to DDR4 (4GB)               │
└─────────────────────────────────────────────────────────────┘
```

### R5F vs A53 — Key Differences

| Feature | Cortex-A53 (APU) | Cortex-R5F (RPU) |
|---------|------------------|-------------------|
| Clock speed | 1.3 GHz | 533 MHz |
| Pipeline | 8-stage, in-order | 8-stage, in-order |
| Memory | Virtual (MMU), Linux | Physical (MPU), bare-metal/RTOS |
| Caches | 32KB L1, 1MB L2 shared | 32KB L1, **128KB TCM** (tightly coupled) |
| Latency | Variable (OS, cache misses) | **Deterministic** — guaranteed worst-case timing |
| FPU | Yes (VFPv4) | Yes (VFPv5, single precision) |
| Modes | Normal Linux multitasking | **Split mode** (2 independent cores) or **Lockstep mode** (redundancy) |
| Best for | General computation, OS | Hard real-time, predictable latency |

### How They Work — Programming Model

The R5F cores run independently from Linux on the A53s. They communicate via:

1. **Shared DDR4 memory** — both APU and RPU can read/write to designated shared memory regions
2. **Inter-Processor Interrupts (IPI)** — hardware signals between APU and RPU
3. **OpenAMP / RPMsg** — Linux framework for APU↔RPU communication (message passing over shared memory)

**Typical setup:**
```
A53 (Linux) ─── RPMsg channel ───► R5F (bare-metal or FreeRTOS)
                │                        │
                ├── Send: "probe TT for hash 0xABCD"
                │                        ├── Read DDR4 TT entry
                │                        ├── Return result
                ◄── Receive: "hit, score=+145, depth=18"
```

### How to Use R5F for Chess — Practical Ideas

#### Idea 1: TT Probe Offloading (Questionable — Verify Latency First)
The intuition: dedicate an R5F core to managing the transposition table, using its 128 KB tightly-coupled memory (TCM) to hold hot entries while DDR4 holds the full table. Frees A53 L2 cache for search-related data.

**The problem:** TT probes from A53 to DDR4 take ~100 ns. RPMsg / IPI round-trip between A53 and R5F is **~30-100 μs** — three to four orders of magnitude slower. Offloading a 100 ns operation across a 30 μs channel is a net loss unless you batch hundreds of probes per request, which breaks the standard search loop's request/response pattern.

**When this could still work:**
- **Bulk prefetch:** A53 sends "I'm about to search subtree at hash X" and R5F warms TCM with predicted-related entries asynchronously
- **Background TT cleanup:** R5F handles aging / eviction in the TT to free A53 cycles

**Recommendation:** **Skip this idea in v1.** Focus R5F effort on Ideas 2 (time management) and 3 (FPGA eval coordination), which genuinely benefit from deterministic timing and async offloading.

#### Idea 2: Time Management Controller
Chess engines must manage time carefully — allocate more time for complex positions, less for simple ones. This involves:
- Monitoring elapsed time with microsecond precision
- Signaling the search to stop when time runs out
- Adjusting time allocation based on position complexity

**Why R5F helps:** Time management requires **hard real-time guarantees**. Linux on the A53 can have scheduling jitter of 1-10ms, which matters when you're running 4 search threads under time pressure. The R5F can guarantee sub-microsecond timing.

**Implementation:**
```
R5F core 1 — time management:
  - A53 sends: "you have 5.2 seconds for this move"
  - R5F monitors hardware timer
  - R5F signals A53 search threads via IPI when time is ~80% spent
  - R5F sends hard stop signal at 95% time
  - No jitter, no missed deadlines
```

#### Idea 3: FPGA Communication Coordinator
The R5F sits between the A53 and FPGA fabric, managing the AXI interface for NNUE eval requests:
```
A53 search thread → R5F → FPGA (NNUE eval) → R5F → A53
```
The R5F can batch multiple eval requests, handle DMA transfers for DDR4 accumulator weights, and pipeline requests to keep the FPGA saturated.

#### Idea 4: Syzygy Probe Coordinator
When the search reaches ≤6 pieces, the R5F can handle SSD reads asynchronously — the A53 continues searching other branches while the R5F waits for the USB/SSD response.

### Recommended R5F Usage for This Project

| Core | Task | Priority | Difficulty |
|------|------|----------|------------|
| **R5F Core 0** | FPGA eval coordinator (Idea 3) — AXI DMA, batched eval requests | High — keeps FPGA pipeline saturated | Medium |
| **R5F Core 1** | Time management (Idea 2) + Syzygy probe coordinator (Idea 4) | Medium — correctness + responsiveness | Low-Medium |
| ~~TT probe offload~~ | ~~See Idea 1~~ | Skipped in v1 — IPC overhead exceeds DDR4 probe latency | — |

### Getting Started with R5F
1. Use **Xilinx Vitis** to program the R5F cores (bare-metal or FreeRTOS)
2. Set up **OpenAMP** on the Linux side for APU↔RPU communication
3. Xilinx provides examples: `xilinx-wiki.atlassian.net` → "OpenAMP" section
4. Start with a simple RPMsg echo test between A53 and R5F
5. Then implement TT probe service as first real workload

### Admissions Value
Using the R5F cores demonstrates:
- **Heterogeneous computing** — three different processor types (A53 + R5F + FPGA) working together
- **Real-time systems knowledge** — bare-metal programming, deterministic timing, IPI
- **Systems architecture** — understanding when to use which processor for which workload
- **Inter-processor communication** — shared memory, message passing, synchronization

This is exactly the kind of systems-level depth that MIT and CMU love to see.

---

## 20. FPGA Inference Pipeline — Architecture & Implementation

This section specifies the hardware-side implementation of the NNUE evaluator. **This is the most novel and admissions-relevant part of the project; treat it as a first-class deliverable, not an afterthought.** Earlier sections covered the *what* (which model, which features, which data); this section covers the *how* on the FPGA itself.

### 20.1 System Block Diagram

```
   ARM A53 (Linux)                    FPGA Fabric (Programmable Logic)
   ──────────────                    ──────────────────────────────────
   ┌──────────────────┐   AXI4-Lite ┌─────────────────────────────────┐
   │ Search Engine    │ ◄─────────► │ NNUE Top-Level Wrapper          │
   │  - alpha-beta    │             │  ┌───────────────────────────┐  │
   │  - movegen       │             │  │ Command FIFO              │  │
   │  - make/unmake   │             │  │ ('update', 'eval', 'reset')│ │
   └──────────────────┘             │  └─────────────┬─────────────┘  │
            ▲                       │                ▼                │
            │ AXI4-Stream (eval)    │  ┌───────────────────────────┐  │
            │                       │  │ Accumulator Update Unit   │  │
            │                       │  │  - Reads W[feat] from BRAM │ │
            │                       │  │  - Adds/subtracts to acc   │ │
            │                       │  └─────────────┬─────────────┘  │
            │                       │                ▼                │
            │                       │  ┌───────────────────────────┐  │
            │                       │  │ Accumulator (UltraRAM)     │ │
            │                       │  │  - 2 × 256 × int16          │ │
            │                       │  │  - Stack of N search plies  │ │
            │                       │  └─────────────┬─────────────┘  │
            │                       │                ▼                │
            │                       │  ┌───────────────────────────┐  │
            │                       │  │ Layer 1-3 MAC Pipeline    │  │
            │                       │  │  - DSP arrays              │ │
            │                       │  │  - ClippedReLU stages       │ │
            │                       │  └─────────────┬─────────────┘  │
            │                       │                ▼                │
            └───────────────────────┤  ┌───────────────────────────┐  │
                                    │  │ Result FIFO (int16 cp)    │  │
                                    │  └───────────────────────────┘  │
                                    └─────────────────────────────────┘
```

### 20.2 Two Operation Modes

The host issues two distinct commands to the NNUE engine:

**`update`** (cheap, called per make_move / unmake_move during search):
- Inputs: a list of features added and removed (typically 2-4 of each)
- Action: incrementally update the accumulator using +/- weight rows from BRAM
- Latency: ~16-32 cycles per add/sub pair (one BRAM read + parallel int16 adds)
- Output: updated accumulator state (kept in UltraRAM, NOT returned to host)

**`eval`** (called only at leaf nodes):
- Inputs: side-to-move flag
- Action: read accumulator, run dense layers 1-3, ClippedReLU between each
- Latency: ~40-80 cycles total
- Output: signed int16 centipawn score returned via AXI

Notice the asymmetry: updates happen at every search node (millions per second) but evals only at leaves (10× less frequent). Optimize update latency first.

### 20.3 Accumulator Update Unit

The accumulator holds `2 × 256 × int16` = 1 KB per ply. Across the search stack (up to ply 64), that's 64 KB — fits comfortably in UltraRAM with room to spare.

```
Per single-feature update:
  addr        = feature_index                              // 10 bits, indexes 768 rows
  weight_row  = BRAM[addr]                                  // 256 × int8 = 256 bytes (one BRAM read)
  for i in 0..255 (parallel):
      acc[i] = acc[i] +/- weight_row[i]                     // int16 += int8
```

This loop unrolls fully: 256 int16 adders/subtractors operating in parallel in one cycle.

**Resource estimate:**
- 256 int16 adders ≈ 6,400 LUTs (~2.5% of the XCK26's 256K logic cells)
- 1 BRAM read port @ 256-bit wide (configure as 8-block × 32-bit-wide BRAM array)

**For the DDR4 variant** (large net), weight rows live in DDR4 instead of BRAM. The update unit becomes an AXI master that reads a single 256-byte row per feature change. DDR4 burst-read latency ~80 ns + 1 cycle int16 add per row ≈ ~100 ns per update. Acceptable, but slower than the BRAM-only design.

### 20.4 Layer 1-3 MAC Pipeline

Layers 1 (512→32), 2 (32→32), and 3 (32→1) are small dense matmuls.

```
Layer 1: 512 inputs × 32 outputs = 16,384 int8×int8 MACs
  Throughput target: 32 MACs/cycle (one DSP slice per output channel)
  Cycles: 512 input cycles + ~10 pipeline drain = ~520 cycles total

Layer 2: 32 × 32 = 1,024 MACs
  Throughput: 32 MACs/cycle
  Cycles: ~40

Layer 3: 32 × 1 = 32 MACs
  Throughput: 32 MACs/cycle  
  Cycles: ~2

Total eval latency: ~600 cycles ≈ 3 μs at 200 MHz
DSP utilization: 96 slices (~8% of 1,248 available)
```

Plenty of headroom. Could parallelize Layer 1 with multiple input lanes for 2-4× throughput if needed — but the eval throughput is rarely the bottleneck (the *update* throughput is, since updates are 10× more frequent).

### 20.5 Memory Layout

**BRAM allocation (648 KB total available on XCK26):**

| Resource | Size | Purpose |
|----------|------|---------|
| Accumulator weights (BRAM-only NNUE) | 192 KB | 768 rows × 256 int8 (shared across perspectives) |
| Layer 1 weights | 16 KB | 512 × 32 int8 |
| Layer 2 weights | 1 KB | 32 × 32 int8 |
| Layer 3 weights | 32 B | 32 × 1 int8 |
| Per-layer biases | < 2 KB | int32 biases per output channel per layer |
| Pipeline buffers | ~4 KB | Intermediate int16 between layers |
| **Total BRAM usage** | **~215 KB** | **~33% of 648 KB available** |

**UltraRAM allocation (2.25 MB total):**

| Resource | Size | Purpose |
|----------|------|---------|
| Accumulator stack | 64 KB | 64 plies × 2 perspectives × 256 int16 |
| Reserved | ~2.18 MB | Future: larger accumulator, DDR4 weight cache |

### 20.6 Quantization & Datatypes

This is the part most likely to have subtle bugs. **Get the bit widths exactly right or your FPGA output won't match your PyTorch reference.**

| Stage | Datatype | Range | Notes |
|-------|----------|-------|-------|
| Input features | binary | {0, 1} | Sparse, one bit per (piece, square) feature |
| Accumulator weights | int8 | [-128, 127] | Per-output-channel scale stored alongside |
| Accumulator state | **int16** | wide enough for ~32 × int8 sums | **MUST be int16, not int8 — see Deep Dive §13.3** |
| ClippedReLU output | uint8 | [0, 127] | Standard NNUE convention; shifts int16 down to uint8 |
| Layer 1-2 weights | int8 | [-128, 127] | Per-row scales |
| Layer 1-2 accumulator | int32 | accumulating 512 × int8 × int8 | Then >> shift + clamp back to uint8 for next layer |
| Final output | int16 | signed centipawn score | |

**Critical sanity check:** With 32 active features and int8 weights up to ±127, the accumulator sum can reach ±4,064. That's well outside int8 range (±127). **int16 holds up to ±32,767**, leaving 8× safety headroom.

### 20.7 Timing Closure Plan

Target: 200 MHz on XCK26 (-1 speed grade). **Start conservative at 100 MHz** to get the design working before chasing frequency.

Strategy:
1. **Get the design working at 100 MHz** with a 4-stage pipeline. Verify functional correctness against the PyTorch reference (see §20.8).
2. **Profile critical path** with Vivado timing reports. Likely candidates:
   - Long add chains in MAC accumulators → pipeline into 2-3 stages
   - BRAM read-to-DSP path → add a pipeline register after BRAM output
3. **Incrementally push to 150 MHz, then 200 MHz** by adding pipeline registers.
4. **Accept 150 MHz if 200 MHz requires excessive logic duplication.** A 25% slower design that works is infinitely better than a 200 MHz design that fails timing.

**Throughput estimates:**
- At 200 MHz: ~330k evals/sec sustained (200M cycles/sec ÷ 600 cycles/eval)
- At 100 MHz: ~165k evals/sec — still faster than ARM A53 can call it

The realistic bottleneck is **AXI command overhead** (each eval request crosses the PS-PL boundary), not eval throughput itself. Plan to batch incremental update commands client-side and only issue eval requests at leaf nodes.

### 20.8 Verification Strategy

1. **Bit-accurate PyTorch reference model.**
   - Implement the int8 forward pass in PyTorch with exact rounding/clamping rules matching the FPGA
   - Generate test vectors: 10,000 random positions, save `(features_stm, features_nstm, expected_score)`
   - Both the FPGA simulation AND the C++ runtime must match this golden reference bit-exactly

2. **Per-module cocotb testbenches:**
   - `test_accumulator_update.py` — add/subtract single features, verify against PyTorch
   - `test_mac_pipeline.py` — verify each layer against PyTorch
   - `test_clipped_relu.py` — verify clamping behavior at boundary values
   - `test_top_level.py` — end-to-end eval matches PyTorch within 1 LSB

3. **Hardware-in-the-loop test:**
   - Stream 1,000 positions from host to FPGA, compare output to PyTorch float32 output
   - Acceptance: 95% of evals within ±2 centipawns of float32 baseline (some loss is inherent to int8 quantization)

4. **Tournament-level regression:**
   - Run the same NNUE on CPU (float32) vs FPGA (int8). Play 200-game match
   - Acceptance: ELO difference within ±30 (statistical noise at 200 games)

### 20.9 HLS vs RTL Decision

**Recommendation: Vitis HLS for the MAC pipeline, hand-written SystemVerilog RTL for control + AXI.**

| Component | HLS or RTL | Why |
|-----------|------------|-----|
| Top-level wrapper, FSM | RTL (SystemVerilog) | Clear control flow, easy to debug |
| AXI4-Lite slave interface | Vivado IP / RTL | Standard pattern, IP-generated |
| Accumulator update unit | RTL | Tight control over BRAM read timing |
| MAC pipeline (Layers 1-3) | Vitis HLS (C++) | Loop-heavy, HLS pipelines well |
| ClippedReLU | RTL (1-line module) | Trivial |

**Why not pure HLS?** HLS produces decent results for dataflow kernels but struggles with mixed control/dataflow designs like this one. Hand-written RTL for the control plane gives predictable timing and easy debug; HLS for the arithmetic-heavy parts gets you 90% of optimal performance with 10% of the effort.

### 20.10 Development Phases (Realistic Timeline)

Assuming 10-15 hrs/week of post-internship time on the FPGA track:

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| 1. Bit-accurate PyTorch reference + 10K test vectors | 1 | Golden model file, JSON test cases |
| 2. ClippedReLU + accumulator update RTL | 1-2 | Modules passing cocotb tests |
| 3. MAC pipeline (HLS) | 2-3 | Single-layer matmul with ClippedReLU |
| 4. Full pipeline integration, BRAM weight loading | 1-2 | Top-level passes all test vectors in sim |
| 5. AXI interface + bitstream build | 1-2 | Loadable `.bit` file, FPGA runs eval |
| 6. Linux driver + C++ runtime integration | 1 | Engine can call FPGA eval from C++ |
| 7. Hardware-in-the-loop validation | 1 | 95% within ±2 cp of float32 reference |
| 8. Timing closure push 100 → 200 MHz | 1-2 | Final synthesis report |
| **Total** | **9-14 weeks** | **Working FPGA NNUE eval on KV260** |

This timeline assumes the BRAM-only NNUE. The DDR4 variant adds **~4-6 weeks** for AXI DMA controller and DDR4 weight cache design. Start the DDR4 variant only AFTER the BRAM design works end-to-end.

### 20.11 Risks Specific to FPGA Work

| Risk | Mitigation |
|------|------------|
| Quantization mismatch between PyTorch and FPGA | Build the bit-accurate PyTorch reference in Phase 1 BEFORE writing RTL. Treat any mismatch as a P0 bug. |
| Vivado synthesis time grows to hours | Use incremental compilation. Split design into logically separate IP blocks. |
| Timing closure fails at 200 MHz | Accept 100-150 MHz. Throughput is rarely the project's binding constraint. |
| AXI command overhead dominates eval latency | Batch update commands client-side. Only roundtrip to FPGA for eval at leaf nodes. |
| Bitstream load fails on KV260 | Use Xilinx's reference PetaLinux BSP. Don't customize bootloader unless required. |

---

*Document updated May 28, 2026. Added: 5-model evaluation matrix (including Stockfish-distilled baseline), Stockfish pre-evaluated data sources, full summer deliverables checklist, post-summer testing matrix with 6 eval configs × 2 platforms.*
*Further updates May 28, 2026 (afternoon): rewrote §12 admissions strategy to lead with research contribution; expanded §15 with realistic hour estimates and 9-month post-internship schedule; added §20 FPGA inference architecture with block diagram, quantization plan, verification strategy, and 9-14 week development timeline; clarified shared-accumulator design and parameter count consistency throughout (~213K, not "200K" or "411K"); added reality-check caveats to §10 ELO progression; softened §19 R5F TT-offload idea after IPC latency analysis; updated §14 with explicit epoch budgets; updated §17 ARM A53 risk row with realistic ELO ceiling.*

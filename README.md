# COMP579 Project: Belief-Guided LLM Agents for Chameleon

## Overview

This project extends a ChatArena-style environment to study **LLM agents in a social deduction setting** (the game *Chameleon*). The focus is on how agents:

- Generate informative yet deceptive natural language clues
- Maintain and update **latent beliefs** about hidden variables
- Act under **partial observability**
- Improve behavior using **reinforcement learning (GRPO-style training)**

The system supports both:
- **Closed-source LLM baselines** (OpenAI, Claude, Gemini)
- **Open-source HuggingFace models** with trainable policies via LoRA

---

## Game Description: Chameleon

Chameleon is a multi-player social deduction game with hidden information.

### Roles
- **Non-chameleons**: know the secret word
- **Chameleon**: does *not* know the word

### Phases
1. **Clue Phase** — each player gives a short clue related to the secret word
2. **Accusation Phase** — players vote for who they think is the chameleon (belief-guided for open-source agents)
3. **Guess Phase** — if caught, the chameleon can still win by guessing the word

### Objective
- Non-chameleons: identify the chameleon without exposing the secret word
- Chameleon: avoid detection or correctly guess the word

---

## Repository Structure

```
COMP579_project/
├── run_experiment.py              # Unified CLI entry point
├── main.py                        # Minimal quickstart script
├── requirements.txt
├── configs/                       # Game config files (3p, 5p, 7p)
├── scripts/                       # SLURM job scripts
├── logs/                          # Experiment outputs (transcripts, summaries)
└── chatarena/
    ├── chatarena/
    │   ├── chameleon_agent.py     # Player agent with belief-based voting
    │   ├── chameleon_arena.py     # ChameleonArena + GRPO training loop + RunLogger
    │   ├── backends/
    │   │   ├── llm.py             # TransformersHuggingFaceChat (generation + scoring)
    │   │   ├── openai.py          # OpenAI API backend
    │   │   ├── anthropic.py       # Claude API backend
    │   │   └── gemini.py          # Gemini API backend
    │   └── environments/
    │       ├── chameleon.py       # Original ChatArena Chameleon env
    │       └── chameleon_grpo.py  # Extended env with belief state + reward
    └── experiments/
        ├── cs_experiment.py       # Closed-source baseline runner
        └── grpo_experiment.py     # Open-source GRPO training runner
```

---

## Architecture

### Core Components

#### 1. `ChameleonArena` (`chameleon_arena.py`)

The central orchestrator for GRPO training runs. It wraps the `Chameleon` environment and manages the per-turn loop:

- During the **clue phase** (non-chameleon players only): generates `clue_number` candidate clues, scores each via the belief-based reward, computes normalized advantages, and applies GRPO policy updates before committing the best clue.
- During the **chameleon's clue turn**: generates a single clue using the frozen base model (LoRA adapter disabled), scored for logging only.
- During the **accusation phase**: non-chameleon players vote using `player_belief`; the chameleon votes randomly.
- During the **guess phase**: the chameleon samples a word from `word_belief`.

The arena also holds a `RunLogger` that writes structured per-step logs including reward breakdowns, GRPO losses, and post-clue belief distributions.

#### 2. `Chameleon` Environment (`environments/chameleon_grpo.py`)

Extends the base `Environment` with a live belief state. At reset, a topic, secret word, and chameleon are randomly selected.

**Belief state** (two tensors, updated after every clue):
- `player_belief` — probability distribution over which player is the chameleon
- `word_belief` — probability distribution over candidate words for the current topic

**Belief update rule**: the HuggingFace backend's cross-encoder scores each `(clue, word)` pair using `BAAI/bge-reranker-v2-m3`. Scores are normalized and used to update both belief tensors via a log-space Bayesian update (`log_p + η·score → softmax`).

#### 3. `Player` Agent (`chameleon_agent.py`)

Wraps an `IntelligenceBackend` and exposes:
- `act(observation)` — queries the backend; chameleon players automatically disable the LoRA adapter so they generate without fine-tuned weights.
- `vote_from_belief(player_belief)` — deterministically picks the highest-probability non-self player.
- `guess_from_belief(word_belief)` — samples a word from the word belief distribution.
- `random_vote()` — fallback used by the chameleon during accusation.

#### 4. `TransformersHuggingFaceChat` Backend (`backends/llm.py`)

A single backend serving dual roles:

- **Generation**: loads a causal LM (default: `Qwen/Qwen2.5-7B-Instruct`) with optional LoRA via PEFT. Returns the generated action string plus token log-probabilities and prompt token IDs needed by the GRPO loss.
- **Scoring**: loads `BAAI/bge-reranker-v2-m3` as a cross-encoder to score `(clue, candidate_word)` pairs for belief updates and reward computation.

LoRA is applied only to `q_proj` and `v_proj` by default (`r=16`, `alpha=32`). During the chameleon's turn, `model.disable_adapter()` is used so the chameleon acts from the frozen base model.

---

### Reward Function

Rewards are computed per candidate clue for non-chameleon players only. The components are:

```
reward = -α · self_suspicion  -  γ · max(word_leak − thresh, 0)  -  length_penalty
```

| Term | Description |
|---|---|
| `self_suspicion` | Increase in `player_belief[speaker]` after the clue — penalises clues that make the speaker look suspicious |
| `word_leak` | Increase in `word_belief[true_word]` — penalises clues that narrow the word down too precisely |
| `length_penalty` | Exponential penalty for clues exceeding `max_tokens`, capped at `length_cap` |

Key hyperparameters (all configurable via CLI):

| Flag | Default | Meaning |
|---|---|---|
| `--reward-alpha` | 0.5 | Self-suspicion coefficient |
| `--reward-gamma` | 2.0 | Word-leak coefficient |
| `--reward-word-leak-threshold` | 0.15 | Leak below this value is ignored |
| `--reward-max-tokens` | 12 | Token budget before length penalty activates |
| `--reward-zeta` | 0.1 | Length penalty rate: `exp(ζ · over_by) − 1` |
| `--reward-length-cap` | 2.0 | Maximum length penalty magnitude |

---

### GRPO Training Loop

For each non-chameleon player turn:
1. Sample `clue_number` candidate clues from the policy (with LoRA enabled).
2. Score each candidate → rewards and advantages (mean-normalized).
3. Run `num_grpo_epochs` gradient steps using the clipped PPO objective plus a KL penalty against the frozen base model (computed by temporarily disabling the LoRA adapter).
4. Commit the highest-advantage clue to the environment.

The GRPO loss per response is:

```
L = -min(r·A, clip(r, 1-ε, 1+ε)·A) / seq_len  +  β · KL(θ ‖ ref) / seq_len
```

where `r = exp(log π_θ − log π_old)` and `β = 0.3`, `ε = 0.2` by default.

---

## Running Experiments

### Entry Point

```bash
python run_experiment.py <mode> [options]
```

| Mode | Description |
|---|---|
| `cs` | Closed-source LLM baseline (no belief updates) |
| `cs-belief` | Closed-source with belief-based voting |
| `os` | Open-source HuggingFace baseline (no GRPO) |
| `grpo` | Open-source with full GRPO training |

### Closed-Source Baseline

```bash
python run_experiment.py cs \
  --config configs/chameleon_closed_3p.json \
  --backend openai \            # openai | claude | gemini
  --num-runs 20 \
  --experiment-id openai-3p \
  --log-dir logs/closed_source/openai-3p \
  --save-transcript
```

### GRPO Training

```bash
python run_experiment.py grpo \
  --config configs/chameleon_closed_3p.json \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device 0 \
  --num-runs 50 \
  --clue-number 8 \
  --experiment-id qwen-grpo-3p \
  --log-dir logs/grpo/qwen-3p \
  --eval-runs 10 \
  --save-transcript
```

### SLURM (HPC)

Pre-written job scripts are in `scripts/`:

```bash
sbatch scripts/cs_experiment.slurm      # closed-source baseline
sbatch scripts/cs_belief.slurm          # closed-source + belief voting
sbatch scripts/grpo_train.slurm         # GRPO open-source training
```

---

## Configuration Files

Game configs live in `configs/` (and mirrored under `chatarena/examples/`). They specify the number of players, per-player role descriptions, and the shared global prompt. Available presets:

| File | Players |
|---|---|
| `chameleon_closed_3p.json` | 3 |
| `chameleon_closed_5p.json` | 5 |
| `chameleon_closed_7p.json` | 7 |

---

## Installation

```bash
pip install -e chatarena/          # install the chatarena package
pip install -r requirements.txt    # torch, transformers, peft, trl, accelerate, bitsandbytes
```

Set your API key environment variables as needed:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

---

## Outputs

Each experiment writes to `--log-dir`:

| File | Contents |
|---|---|
| `<id>.log` | Per-step console log (rewards, advantages, GRPO losses, beliefs) |
| `<id>_summary.txt` | Win-rate statistics and per-run breakdown |
| `<id>_transcript.txt` | Full message transcript (if `--save-transcript`) |
| `<id>_arena.log` | Detailed arena log including reward weights (GRPO mode) |

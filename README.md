# COMP579 Project: Belief-Guided LLM Agents for Chameleon

## Overview
This project extends a ChatArena-style environment to study **LLM agents in a social deduction setting** (the game *Chameleon*). The focus is on how agents:

- Generate informative yet deceptive natural language clues  
- Maintain and update **latent beliefs** about hidden variables  
- Act under **partial observability**  
- Improve behavior using **reinforcement learning (GRPO-style training)**  

The system supports both:
- **Closed-source LLM baselines** (OpenAI, Claude, Gemini)
- **Open-source HuggingFace models** with trainable policies

---

## Game Description: Chameleon

Chameleon is a multi-player social deduction game with hidden information.

### Roles
- **Non-chameleons**: know the secret word
- **Chameleon**: does *not* know the word

### Phases
1. **Clue Phase**
   - Each player gives a short clue related to the secret word
2. **Accusation Phase**
   - Players vote for who they think is the chameleon
3. **Guess Phase**
   - If caught, the chameleon can still win by guessing the word

### Objective
- Non-chameleons: identify the chameleon  
- Chameleon: avoid detection or correctly guess the word  

---

## Key Features

### 1. Multi-Agent LLM Environment
- Fully simulated multi-player environment
- Configurable via JSON (e.g., `chameleon_closed_3p.json`)
- Supports multiple LLM backends

---

### 2. Belief Modeling

Each agent maintains **probabilistic beliefs**:

- Non-chameleons:
  - Distribution over which player is the chameleon
- Chameleon:
  - Distribution over possible secret words

Beliefs are updated after each clue using learned representations.

#### Architecture
- GRU-based recurrent belief updates
- Shared embeddings:
  - Speaker embeddings
  - Topic embeddings
- Separate heads:
  - Player belief head
  - Word belief head

---

### 3. GRPO-Based Training

Open-source agents are trained using a **policy gradient method** similar to GRPO.

#### Key components:
- Sample multiple candidate clues per turn
- Score each candidate using a belief-based reward
- Compute normalized advantages
- Apply policy update using:
  - Clipped objective
  - KL regularization vs reference model

Training is performed using HuggingFace Transformers + LoRA.

---

### 4. Belief-Based Reward

The reward signal is designed to encourage:

- For non-chameleons:
  - Lower suspicion from other agents
  - Avoid revealing the secret word too clearly
- For chameleon:
  - Blend in with consistent clues
  - Improve word inference

Rewards are computed from **changes in belief distributions**.

---

### 5. HuggingFace Backend

Custom backend supports:

- Chat-style prompting
- Token-level log probabilities
- Sequence log probabilities
- Sentence embeddings
- LoRA fine-tuning

This enables:
- GRPO loss computation
- Fine-grained control over generation

---

## Running Experiments

### Entry Point

```bash
python run_experiment.py <mode>

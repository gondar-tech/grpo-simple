# GRPO Training Guide: Teaching Qwen-3-0.6B to Play Liar's Dice & Gin Rummy

> **Stack:** `Qwen/Qwen3-0.6B-Instruct` · HuggingFace TRL `GRPOTrainer` · LoRA (PEFT) · PyTorch

---

## Table of Contents
1. [What is GRPO? (Plain English)](#1-what-is-grpo)
2. [Architecture Overview](#2-architecture-overview)
3. [Why These Games Are Good GRPO Targets](#3-why-these-games)
4. [Environment Setup](#4-environment-setup)
5. [Step 1 — Model & Tokenizer Loading](#5-step-1--model--tokenizer-loading)
6. [Step 2 — Game Environment (Liar's Dice focus)](#6-step-2--game-environment)
7. [Step 3 — Prompt Dataset Generation](#7-step-3--prompt-dataset-generation)
8. [Step 4 — Reward Functions](#8-step-4--reward-functions)
9. [Step 5 — GRPO Training Config & Trainer](#9-step-5--grpo-training-config--trainer)
10. [Step 6 — Running & Monitoring Training](#10-step-6--running--monitoring-training)
11. [Step 7 — Evaluation & Testing](#11-step-7--evaluation--testing)
12. [Gin Rummy Variant Notes](#12-gin-rummy-variant-notes)
13. [Tips, Gotchas & Next Steps](#13-tips-gotchas--next-steps)
14. [Reference Videos & Articles](#14-reference-videos--articles)

---

## 1. What is GRPO?

**GRPO (Group Relative Policy Optimization)** was introduced by DeepSeek to train reasoning models
(like DeepSeek-R1) more efficiently than the older PPO algorithm.

The key idea in one sentence:
> *Generate N responses to the same prompt, score each with a reward function, and nudge the model
> toward responses that scored above average.*

**Why is this great for games?**
- Games have **verifiable outcomes** (valid move? win/lose?) — perfect for rule-based reward functions.
- No human labelers needed.
- Qwen-3-0.6B is tiny enough to train on a single consumer GPU (8–16 GB VRAM with LoRA).

**GRPO vs PPO at a glance:**

| Feature | PPO | GRPO |
|---|---|---|
| Needs Critic model? | ✅ Yes (expensive) | ❌ No |
| Needs Reward model? | ✅ Yes (RLHF) | ❌ No (rule-based OK) |
| Memory cost | High | Low |
| Good for games? | Possible | ✅ Ideal |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                   │
│                                                         │
│  Game State  ──►  Prompt  ──►  Qwen-3-0.6B (Policy)     │
│                               │                         │
│                         Generate N=8                    │
│                         completions                     │
│                               │                         │
│              ┌────────────────┴──────────────────┐      │
│              │  Reward Functions (Rule-Based)    │      │
│              │  1. valid_move_reward()           │      │
│              │  2. format_reward()               │      │
│              │  3. strategy_reward()             │      │
│              └────────────────┬──────────────────┘      │
│                               │                         │
│                    Scores: [0.9, 0.2, 1.0, ...]         │
│                               │                         │
│                    Z-score normalize → Advantages       │
│                               │                         │
│              ┌────────────────┴──────────────────┐      │
│              │  GRPO Loss (clipped policy ratio  │      │
│              │  + KL divergence penalty)         │      │
│              └────────────────┬──────────────────┘      │
│                               │                         │
│                    Update model weights (LoRA)          │
└─────────────────────────────────────────────────────────┘
```

**Components you will build:**
- `LiarsDiceEnv` — encodes the game state into a text prompt
- `generate_dataset()` — creates thousands of game situation prompts
- `valid_move_reward()` — checks if the LLM's chosen action is legal
- `format_reward()` — checks if the output follows `<think>...</think><action>...</action>`
- `strategy_reward()` — rewards winning game outcomes

---

## 3. Why These Games Are Good GRPO Targets

**Liar's Dice** (chosen as primary example):
- Each turn has a small, **enumerable action space** (bid or challenge)
- Outcomes are binary and **fully verifiable** (valid bid? won the challenge?)
- Natural language is ideal: "I bid 3 fives" is easy for an LLM to parse
- Bluffing = reasoning under uncertainty — great for chain-of-thought

**Gin Rummy** (secondary example — notes in Section 12):
- Larger action space (draw/discard from 52 cards)
- Reward is score-based (deadwood points)
- Requires tracking sets and runs — tests structured reasoning

---

## 4. Environment Setup

```bash
# Create a clean Python environment (Python 3.10+ recommended)
conda create -n grpo-game python=3.10 -y
conda activate grpo-game

# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hugging Face ecosystem
pip install transformers==4.51.0   # Qwen3 support requires recent version
pip install trl>=0.15.0            # GRPOTrainer lives here
pip install peft>=0.14.0           # LoRA adapters
pip install accelerate>=0.34.0     # multi-GPU / mixed precision
pip install datasets>=2.20.0       # HuggingFace datasets

# Utilities
pip install wandb   # experiment tracking (optional but recommended)
pip install numpy pandas
```

**Minimum hardware:**
- GPU: 8 GB VRAM (e.g., RTX 3080) — works with LoRA + bfloat16
- Recommended: 16 GB VRAM for `num_generations=8`
- CPU fallback: very slow, use `num_generations=2` and `max_completion_length=64`

---

## 5. Step 1 — Model & Tokenizer Loading

```python
# ── file: train_grpo_game.py ─────────────────────────────────────────────────

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ── 1a. Model ID ─────────────────────────────────────────────────────────────
# We use Qwen3-0.6B-Instruct: small enough to train on a single GPU,
# but capable of following structured instructions.
MODEL_ID = "Qwen/Qwen3-0.6B-Instruct"

# ── 1b. Load the tokenizer ────────────────────────────────────────────────────
# The tokenizer converts text ↔ token IDs.
# trust_remote_code=True is required for Qwen's custom tokenizer class.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="left",   # Left-padding is standard for causal LM generation
)
# Qwen3 uses a special end-of-sequence token; make sure pad = eos
tokenizer.pad_token = tokenizer.eos_token

# ── 1c. Load the base model in bfloat16 ──────────────────────────────────────
# bfloat16 halves memory vs float32 with minimal accuracy loss.
# device_map="auto" spreads layers across available GPUs/CPU automatically.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,   # Use bfloat16 for memory efficiency
    device_map="auto",            # Automatically place on GPU(s)
    trust_remote_code=True,
)

# ── 1d. Configure LoRA ────────────────────────────────────────────────────────
# LoRA (Low-Rank Adaptation) adds small trainable matrices to certain layers.
# This means we only train ~1-2% of parameters instead of all 600M weights.
# r=16 is the rank — higher = more capacity but more memory.
# target_modules are the attention projection matrices inside the transformer.
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,                          # LoRA rank (8–64 range typical)
    lora_alpha=32,                 # Scaling factor: effective lr = alpha/r * lr
    lora_dropout=0.05,             # Light dropout for regularization
    target_modules=[               # Apply LoRA to attention layers only
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # MLP layers too for Qwen3
    ],
    bias="none",                   # Don't train biases (saves memory)
)

# Wrap the model with LoRA adapters.
# Only the LoRA parameters will be updated during training.
model = get_peft_model(model, lora_config)

# Print a summary: should show ~1-3% trainable parameters
model.print_trainable_parameters()
# Expected output: "trainable params: 8,257,536 || all params: 604,123,136 || trainable%: 1.37"
```

---

## 6. Step 2 — Game Environment

```python
# ── file: game_env.py ─────────────────────────────────────────────────────────

import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# LIAR'S DICE RULES (simplified 2-player version)
# ─────────────────────────────────────────────────────────────────────────────
# Each player starts with 5 dice. Players take turns bidding on how many dice
# of a given face value (1–6) are on the table in total (both players' dice).
# A player can:
#   - BID: claim a higher quantity OR same quantity with higher face value
#   - CHALLENGE (call "Liar!"): if you think the last bid is false
# If challenged, all dice are revealed. If the bid is true → challenger loses a die.
# If bid is false → bidder loses a die. Game ends when one player has 0 dice.

@dataclass
class LiarsDiceState:
    """Represents a single game state (one player's turn)."""

    # The LLM-controlled player's dice (hidden from opponent)
    my_dice: list[int] = field(default_factory=list)

    # The opponent's dice count (we don't know their values)
    opponent_dice_count: int = 5

    # The current bid on the table: (quantity, face_value) or None for first move
    current_bid: Optional[Tuple[int, int]] = None

    # Whose turn it is: "llm" or "opponent"
    current_turn: str = "llm"

    # Track dice counts
    my_dice_count: int = 5


class LiarsDiceEnv:
    """
    Manages a Liar's Dice game and converts game states to text prompts
    that the LLM can understand and respond to.
    """

    def __init__(self, num_dice: int = 5):
        self.num_dice = num_dice
        self.reset()

    def reset(self) -> LiarsDiceState:
        """Start a new game: roll fresh dice for both players."""
        # Roll dice: values 1–6 for each die
        my_dice = [random.randint(1, 6) for _ in range(self.num_dice)]
        self.state = LiarsDiceState(
            my_dice=my_dice,
            opponent_dice_count=self.num_dice,
            current_bid=None,
            my_dice_count=self.num_dice,
        )
        return self.state

    def state_to_prompt(self, state: LiarsDiceState) -> str:
        """
        Convert a game state into a text prompt for the LLM.
        The quality of this prompt directly affects how well the model learns!
        We include:
          - The rules (for the model to reason about)
          - Current dice (the model's private info)
          - Current bid on table
          - Valid actions
        """
        # Describe dice as a readable summary
        dice_counts = {i: state.my_dice.count(i) for i in range(1, 7)}
        dice_desc = ", ".join(
            f"{count}x {face}" for face, count in dice_counts.items() if count > 0
        )

        # Describe the current bid
        if state.current_bid is None:
            bid_desc = "No bid yet — you go first."
            valid_actions = "You must make the opening bid."
        else:
            qty, face = state.current_bid
            bid_desc = f"Current bid: {qty} dice showing {face}s"
            valid_actions = (
                f"You can: (1) bid higher — either more than {qty} of any face, "
                f"or {qty}+ of a face > {face}, "
                f"(2) challenge — call the opponent a liar."
            )

        total_dice = state.my_dice_count + state.opponent_dice_count

        prompt = f"""You are playing Liar's Dice. Here is the current situation:

YOUR DICE ({state.my_dice_count} dice): {dice_desc}
OPPONENT'S DICE: {state.opponent_dice_count} (values hidden)
TOTAL DICE ON TABLE: {total_dice}
{bid_desc}

{valid_actions}

Think step by step about:
1. What dice do I have?
2. What is the probability the bid is true given all {total_dice} dice?
3. Should I bid higher or challenge?

Respond ONLY in this exact format:
<think>
[your reasoning here]
</think>
<action>
[BID quantity face_value] or [CHALLENGE]
Example: BID 3 4  (bidding 3 fours)
Example: CHALLENGE
</action>"""
        return prompt

    def parse_action(self, response: str) -> Optional[Tuple[str, Optional[Tuple[int, int]]]]:
        """
        Parse the LLM's text response into a structured action.
        Returns: ("BID", (quantity, face)) or ("CHALLENGE", None) or ("INVALID", None)
        """
        import re

        # Extract content inside <action> tags
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL)
        if not action_match:
            return ("INVALID", None)  # No action tag found

        action_text = action_match.group(1).strip().upper()

        if action_text == "CHALLENGE":
            return ("CHALLENGE", None)

        # Try to parse BID quantity face_value
        bid_match = re.match(r"BID\s+(\d+)\s+(\d+)", action_text)
        if bid_match:
            qty = int(bid_match.group(1))
            face = int(bid_match.group(2))
            if 1 <= face <= 6 and qty >= 1:  # Basic validity
                return ("BID", (qty, face))

        return ("INVALID", None)  # Could not parse

    def is_valid_action(
        self,
        action: Tuple[str, Optional[Tuple[int, int]]],
        state: LiarsDiceState
    ) -> bool:
        """
        Check if a parsed action is valid given the current game state.
        This is the core of our verifiable reward!
        """
        action_type, bid_data = action

        if action_type == "INVALID":
            return False  # Malformed response

        if action_type == "CHALLENGE":
            # You can always challenge if there's an existing bid
            return state.current_bid is not None

        if action_type == "BID":
            qty, face = bid_data
            if state.current_bid is None:
                # Opening bid: anything goes (qty ≥ 1, face 1–6)
                return True
            prev_qty, prev_face = state.current_bid
            # Valid bid: higher quantity, OR same quantity with higher face
            return (qty > prev_qty) or (qty == prev_qty and face > prev_face)

        return False

    def evaluate_challenge(self, state: LiarsDiceState, opponent_dice: list[int]) -> bool:
        """
        Resolve a CHALLENGE. Returns True if the challenger (LLM) wins.
        In a real game, both players reveal dice.
        For training, we simulate the opponent's dice.
        """
        if state.current_bid is None:
            return True  # Can't challenge with no bid

        qty, face = state.current_bid
        total_dice = state.my_dice + opponent_dice
        actual_count = total_dice.count(face)

        # Challenger wins if the actual count is LESS than the bid
        return actual_count < qty
```

---

## 7. Step 3 — Prompt Dataset Generation

```python
# ── file: generate_dataset.py ─────────────────────────────────────────────────

import random
from datasets import Dataset
from game_env import LiarsDiceEnv, LiarsDiceState

def generate_liars_dice_prompts(num_samples: int = 2000) -> Dataset:
    """
    Generate a diverse dataset of Liar's Dice game situations.

    GRPO doesn't need labeled answers — it only needs PROMPTS.
    The model generates its own responses, and we score them with reward functions.
    
    We want diversity:
      - Early game (lots of dice, no bid yet)
      - Mid game (ongoing bids)
      - Late game (few dice, bluffing situations)
    """
    env = LiarsDiceEnv()
    prompts = []

    for i in range(num_samples):
        # Randomly vary the game state for diversity
        my_dice_count = random.randint(1, 5)
        opp_dice_count = random.randint(1, 5)

        # Roll random dice
        my_dice = [random.randint(1, 6) for _ in range(my_dice_count)]

        # Randomly decide if there's an existing bid
        if random.random() < 0.7:
            # Mid-game: there's already a bid
            total = my_dice_count + opp_dice_count
            # Generate a plausible (but not always optimal) bid
            bid_qty = random.randint(1, max(1, total // 2 + 1))
            bid_face = random.randint(1, 6)
            current_bid = (bid_qty, bid_face)
        else:
            # Opening move
            current_bid = None

        state = LiarsDiceState(
            my_dice=my_dice,
            opponent_dice_count=opp_dice_count,
            current_bid=current_bid,
            my_dice_count=my_dice_count,
        )

        # Convert state to the chat format GRPOTrainer expects:
        # A list of message dicts (like a chat conversation)
        prompt_text = env.state_to_prompt(state)

        # Store both the formatted prompt AND the raw state for reward computation
        prompts.append({
            "prompt": [
                {"role": "system", "content": "You are an expert Liar's Dice player."},
                {"role": "user",   "content": prompt_text},
            ],
            # Store metadata as strings (HuggingFace Dataset requires serializable types)
            "my_dice": str(my_dice),
            "current_bid": str(current_bid),
            "my_dice_count": my_dice_count,
            "opp_dice_count": opp_dice_count,
        })

    # Shuffle for training variety
    random.shuffle(prompts)

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list(prompts)

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    return dataset


# Generate the dataset
print("Generating game prompts...")
dataset = generate_liars_dice_prompts(num_samples=3000)
print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")
# Save to disk so we don't regenerate every run
dataset.save_to_disk("liars_dice_dataset")
```

---

## 8. Step 4 — Reward Functions

```python
# ── file: reward_functions.py ─────────────────────────────────────────────────
# This is the MOST IMPORTANT part of GRPO.
# Reward functions are the "teacher" — they tell the model what "good" looks like.
# We use multiple reward functions and sum them (TRL supports this natively).

import re
import ast
import random
from typing import Optional
from game_env import LiarsDiceEnv, LiarsDiceState


# ── Reward 1: Format reward ───────────────────────────────────────────────────
# Reward the model for following the required output format.
# This is crucial: if the format is wrong, we can't parse the action at all.
# Start with format rewards so the model first learns HOW to respond.

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Returns +0.5 if the response has both <think> and <action> tags,
    +0.25 for just one tag, 0.0 if neither.
    
    completions: list of model-generated strings (one per sample in the batch)
    **kwargs: additional columns from the dataset row (my_dice, current_bid, etc.)
    """
    rewards = []
    for completion in completions:
        has_think  = bool(re.search(r"<think>.*?</think>",   completion, re.DOTALL))
        has_action = bool(re.search(r"<action>.*?</action>", completion, re.DOTALL))

        if has_think and has_action:
            rewards.append(0.5)   # Full format reward
        elif has_think or has_action:
            rewards.append(0.25)  # Partial credit
        else:
            rewards.append(0.0)   # No format at all
    return rewards


# ── Reward 2: Valid move reward ───────────────────────────────────────────────
# The core "game rules" reward. Did the model make a legal move?
# This is a VERIFIABLE reward — no ambiguity, no need for human judges.

def valid_move_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Returns +1.0 for a valid game action, 0.0 for invalid.
    
    The kwargs will contain columns from our dataset:
      - current_bid: string representation of (qty, face) tuple or "None"
      - my_dice_count: int
    """
    rewards = []
    env = LiarsDiceEnv()

    for i, completion in enumerate(completions):
        # Reconstruct the game state from the stored metadata
        current_bid_str = kwargs.get("current_bid", ["None"] * len(completions))[i]
        my_dice_count   = kwargs.get("my_dice_count", [5] * len(completions))[i]
        opp_dice_count  = kwargs.get("opp_dice_count", [5] * len(completions))[i]

        # Parse the stored bid back from string
        try:
            current_bid = ast.literal_eval(current_bid_str)  # "(3, 4)" → (3, 4)
        except (ValueError, SyntaxError):
            current_bid = None

        state = LiarsDiceState(
            my_dice=[],       # Not needed for move validation
            current_bid=current_bid,
            my_dice_count=int(my_dice_count),
            opponent_dice_count=int(opp_dice_count),
        )

        # Parse the LLM's action
        action = env.parse_action(completion)

        # Check if it's a valid game action
        if env.is_valid_action(action, state):
            rewards.append(1.0)   # Legal move!
        else:
            rewards.append(0.0)   # Illegal move or malformed response

    return rewards


# ── Reward 3: Strategy reward ─────────────────────────────────────────────────
# Reward good strategic thinking. This is more subjective but still rule-based.
# 
# Strategy heuristics for Liar's Dice:
#   - If you have many of a face value, bidding that face is smart
#   - Challenging a bid that seems impossible given your dice is smart
#   - Bidding extremely high quantities is risky (penalize)

def strategy_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Returns a float in [-0.5, +0.5] based on strategic quality.
    Positive = strategically sound, negative = poor strategy.
    """
    rewards = []
    env = LiarsDiceEnv()

    for i, completion in enumerate(completions):
        # Get dice information
        my_dice_str   = kwargs.get("my_dice", ["[]"]   * len(completions))[i]
        current_bid_str = kwargs.get("current_bid", ["None"] * len(completions))[i]
        my_dice_count = int(kwargs.get("my_dice_count", [5] * len(completions))[i])
        opp_dice_count = int(kwargs.get("opp_dice_count", [5] * len(completions))[i])

        try:
            my_dice     = ast.literal_eval(my_dice_str)
            current_bid = ast.literal_eval(current_bid_str)
        except (ValueError, SyntaxError):
            rewards.append(0.0)
            continue

        action = env.parse_action(completion)
        action_type, bid_data = action

        if action_type == "INVALID":
            rewards.append(-0.5)   # Penalize malformed responses
            continue

        total_dice = my_dice_count + opp_dice_count
        reward = 0.0

        if action_type == "BID" and bid_data:
            qty, face = bid_data
            # How many of this face do I personally have?
            my_count_of_face = my_dice.count(face) if my_dice else 0

            # Heuristic: good bid if my dice support it
            # Expected count in total = (total_dice * 1/6) + my_dice bias
            expected_total = total_dice / 6 + my_count_of_face * 0.5
            if qty <= expected_total:
                reward += 0.3   # Conservative, statistically sound bid
            elif qty <= expected_total * 1.5:
                reward += 0.1   # Slightly risky bluff
            else:
                reward -= 0.3   # Very risky overbid

        elif action_type == "CHALLENGE" and current_bid:
            qty, face = current_bid
            my_count_of_face = my_dice.count(face) if my_dice else 0
            # Expected count if evenly distributed
            expected_total = total_dice / 6 + my_count_of_face * 0.5
            if qty > expected_total * 1.3:
                reward += 0.5   # Smart challenge — bid seems too high
            elif qty < expected_total * 0.7:
                reward -= 0.3   # Unwise challenge — bid seems low/reasonable

        rewards.append(max(-0.5, min(0.5, reward)))  # Clamp to [-0.5, 0.5]

    return rewards


# ── Reward 4: Reasoning quality reward ────────────────────────────────────────
# Reward the model for including substantive reasoning in <think> tags.
# This encourages chain-of-thought behavior (like DeepSeek-R1).

def reasoning_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Returns +0.3 if <think> block mentions relevant concepts (dice, probability, etc.)
    Returns 0.0 if <think> is present but empty/trivial
    Returns -0.1 if no <think> tag at all
    """
    rewards = []
    # Keywords we want to see in the reasoning
    reasoning_keywords = [
        "dice", "probability", "chance", "bid", "challenge",
        "bluff", "risk", "count", "face", "likely", "unlikely",
    ]

    for completion in completions:
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)

        if not think_match:
            rewards.append(-0.1)   # No reasoning at all
            continue

        think_text = think_match.group(1).lower()

        # Count how many reasoning keywords appear
        keyword_hits = sum(1 for kw in reasoning_keywords if kw in think_text)

        if len(think_text.strip()) < 20:
            rewards.append(0.0)    # Reasoning exists but is trivially short
        elif keyword_hits >= 3:
            rewards.append(0.3)    # Good, substantive reasoning
        elif keyword_hits >= 1:
            rewards.append(0.15)   # Some relevant reasoning
        else:
            rewards.append(0.05)   # Reasoning present but not very relevant

    return rewards
```

---

## 9. Step 5 — GRPO Training Config & Trainer

```python
# ── file: train_grpo_game.py (continued) ──────────────────────────────────────

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from reward_functions import (
    format_reward,
    valid_move_reward,
    strategy_reward,
    reasoning_reward,
)

# ── Load dataset ──────────────────────────────────────────────────────────────
dataset = load_from_disk("liars_dice_dataset")
train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

# ── GRPO Training Configuration ───────────────────────────────────────────────
# These hyperparameters are tuned for Qwen3-0.6B on a single 16GB GPU.
# Adjust num_generations down if you get OOM errors.

training_args = GRPOConfig(
    # ── Output ────────────────────────────────────────────────────────────────
    output_dir="qwen3-0.6b-liars-dice-grpo",

    # ── Core GRPO parameters ──────────────────────────────────────────────────
    # num_generations: how many responses to sample per prompt
    # More = better advantage estimates but more memory. Start with 4, try 8.
    num_generations=8,

    # max_completion_length: max tokens the model generates per response
    # Enough for <think>reasoning</think><action>BID 3 4</action>
    max_completion_length=256,

    # max_prompt_length: max tokens for the game state prompt
    max_prompt_length=512,

    # ── Optimization ──────────────────────────────────────────────────────────
    learning_rate=5e-6,               # Low LR for stable RL fine-tuning
    per_device_train_batch_size=2,     # Prompts per device per step (not completions!)
    gradient_accumulation_steps=4,    # Effective batch = 2 * 4 = 8 prompts
    num_train_epochs=3,                # 3 epochs over the dataset

    # ── Mixed precision ────────────────────────────────────────────────────────
    bf16=True,    # Use bfloat16 (requires Ampere GPU or newer: RTX 30xx+, A100)
    fp16=False,   # Don't use fp16 alongside bf16

    # ── KL divergence penalty ─────────────────────────────────────────────────
    # Prevents the model from drifting too far from the original pretrained model.
    # Too high = no learning. Too low = instability. 0.01–0.1 is typical.
    kl_coef=0.04,

    # ── Clipping ──────────────────────────────────────────────────────────────
    # Clips the policy ratio to [1-epsilon, 1+epsilon] for stable updates.
    # This is the "Proximal" part of GRPO (inherited from PPO).
    cliprange=0.2,

    # ── Logging and saving ────────────────────────────────────────────────────
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
    report_to="wandb",        # Change to "none" if not using wandb
    run_name="liars-dice-grpo-v1",

    # ── Qwen3-specific: disable thinking mode for generation ──────────────────
    # Qwen3 has a built-in "thinking" mode that adds overhead.
    # We handle reasoning ourselves via <think> tags in prompts.
    chat_template_kwargs={"enable_thinking": False},
)

# ── Initialize the GRPOTrainer ────────────────────────────────────────────────
# Pass ALL reward functions as a list — TRL sums them automatically.
# You can also weight them: reward_weights=[1.0, 2.0, 0.5, 0.3]
trainer = GRPOTrainer(
    model=model,                       # The LoRA-wrapped Qwen3 model
    processing_class=tokenizer,        # Tokenizer for encoding/decoding
    args=training_args,

    # Reward functions: each takes (completions, **kwargs) and returns list[float]
    reward_funcs=[
        format_reward,      # Weight: 1.0 (format compliance)
        valid_move_reward,  # Weight: 1.0 (game rule validity)
        strategy_reward,    # Weight: 1.0 (strategic quality)
        reasoning_reward,   # Weight: 1.0 (chain-of-thought quality)
    ],

    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

---

## 10. Step 6 — Running & Monitoring Training

```python
# ── file: train_grpo_game.py (final section) ──────────────────────────────────

import wandb

# ── Optional: initialize wandb for experiment tracking ────────────────────────
# wandb gives you live plots of reward, loss, KL divergence, etc.
# If you don't have wandb, set report_to="none" in GRPOConfig above.
wandb.init(
    project="grpo-game-ai",
    name="liars-dice-qwen3-0.6b",
    config={
        "model": MODEL_ID,
        "game": "liars_dice",
        "num_generations": 8,
        "lora_r": 16,
    }
)

# ── Start training! ────────────────────────────────────────────────────────────
# This is the main training loop. TRL handles:
#   1. Generating N completions per prompt
#   2. Calling your reward functions
#   3. Computing advantages via Z-score normalization
#   4. Computing GRPO loss (clipped policy ratio + KL penalty)
#   5. Backpropagating through the LoRA layers
#   6. Logging metrics

print("Starting GRPO training...")
train_result = trainer.train()

# ── Save the trained LoRA adapter ────────────────────────────────────────────
# This saves only the small LoRA delta weights, not the full model.
trainer.save_model("qwen3-0.6b-liars-dice-grpo/final")
tokenizer.save_pretrained("qwen3-0.6b-liars-dice-grpo/final")

print("Training complete! Model saved.")
print(f"Final metrics: {train_result.metrics}")

# ── What to watch in the logs ─────────────────────────────────────────────────
# Key metrics to monitor:
#
#   reward/mean          → Should gradually increase (model getting better)
#   reward/valid_move    → Should approach 1.0 quickly (model learns valid moves first)
#   reward/format        → Should approach max (0.5) quickly
#   kl                   → Should stay < 1.0; spikes indicate instability
#   loss                 → Can fluctuate; overall trend should stabilize
#   grad_norm            → Should stay < 1.0 with max_grad_norm=0.1

# ── Training duration estimates (rough) ───────────────────────────────────────
# RTX 3080 (10GB): ~3–4 hours for 3000 samples × 3 epochs
# RTX 4090 (24GB): ~1–2 hours
# A100 (40GB):    ~30–45 minutes
```

---

## 11. Step 7 — Evaluation & Testing

```python
# ── file: evaluate.py ─────────────────────────────────────────────────────────

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from game_env import LiarsDiceEnv

# ── Load the trained model ────────────────────────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
# Load LoRA adapter on top of the base model
model = PeftModel.from_pretrained(base_model, "qwen3-0.6b-liars-dice-grpo/final")
model = model.merge_and_unload()   # Merge LoRA weights into base for faster inference

tokenizer = AutoTokenizer.from_pretrained("qwen3-0.6b-liars-dice-grpo/final")

def play_one_turn(state, verbose=True):
    """Run the trained model on a game state and return its action."""
    env = LiarsDiceEnv()
    prompt_text = env.state_to_prompt(state)

    # Format as a chat message
    messages = [
        {"role": "system",  "content": "You are an expert Liar's Dice player."},
        {"role": "user",    "content": prompt_text},
    ]

    # Apply the chat template (Qwen3 uses a specific format)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,       # Some randomness for varied play
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],   # Only decode newly generated tokens
        skip_special_tokens=True,
    )

    if verbose:
        print("=== Model Response ===")
        print(response)
        print("=== Parsed Action ===")
        action = env.parse_action(response)
        print(f"Action: {action}")
        valid = env.is_valid_action(action, state)
        print(f"Valid: {valid}")

    return response


# ── Quick evaluation: % of valid moves over N random states ──────────────────
def evaluate_validity(num_trials: int = 100) -> dict:
    """
    Run the model on N random game states and measure:
      - % of valid moves (primary metric)
      - % with correct format
      - Action type distribution (bid vs challenge)
    """
    import random
    env = LiarsDiceEnv()

    valid_count = 0
    format_count = 0
    action_types = {"BID": 0, "CHALLENGE": 0, "INVALID": 0}

    for _ in range(num_trials):
        env.reset()
        # Random mid-game state
        state = env.state
        state.current_bid = (random.randint(1, 3), random.randint(1, 6)) if random.random() > 0.3 else None

        response = play_one_turn(state, verbose=False)
        action = env.parse_action(response)
        action_type = action[0]
        action_types[action_type] += 1

        if env.is_valid_action(action, state):
            valid_count += 1

        import re
        if re.search(r"<think>.*?</think>.*<action>.*?</action>", response, re.DOTALL):
            format_count += 1

    return {
        "valid_move_rate": valid_count / num_trials,
        "format_compliance": format_count / num_trials,
        "action_distribution": {k: v/num_trials for k, v in action_types.items()},
    }


# Run evaluation
print("\n=== Evaluating trained model ===")
results = evaluate_validity(100)
print(f"Valid move rate:    {results['valid_move_rate']:.1%}")
print(f"Format compliance: {results['format_compliance']:.1%}")
print(f"Action mix:        {results['action_distribution']}")

# A well-trained model should achieve:
#   valid_move_rate > 90%
#   format_compliance > 95%
```

---

## 12. Gin Rummy Variant Notes

Gin Rummy is more complex but the same GRPO framework applies. Here's the key difference:

```python
# ── Gin Rummy: Key Design Differences ────────────────────────────────────────

# Action space is larger: draw from stock or discard pile, then discard a card
# State includes: hand (10 cards), discard pile top, known opponent discards

GIN_RUMMY_SYSTEM_PROMPT = """You are a Gin Rummy expert. Your goal is to minimize deadwood 
(unmatched card points) and form melds (sets of 3+ same rank, or runs of 3+ same suit)."""

# Example state-to-prompt for Gin Rummy
def gin_state_to_prompt(hand: list, discard_top: str, turn: int) -> str:
    hand_str = ", ".join(hand)
    return f"""TURN {turn}
YOUR HAND: {hand_str}
TOP OF DISCARD PILE: {discard_top}

Analyze your hand for melds (sets/runs) and deadwood (unmatched cards).
Decide: DRAW_STOCK, DRAW_DISCARD, then which card to DISCARD.
Also consider: GIN (all cards in melds) or KNOCK (≤10 deadwood).

Respond in format:
<think>[reasoning about melds and strategy]</think>
<action>DRAW_STOCK|DRAW_DISCARD DISCARD card_name</action>
Example: <action>DRAW_STOCK DISCARD 7H</action>"""


# Reward functions for Gin Rummy:
def gin_deadwood_reward(completions, **kwargs):
    """Reward lower deadwood scores after the action."""
    # Parse the action, simulate it, calculate deadwood reduction
    # +1.0 for GIN, +0.5 for KNOCK, proportional reward for deadwood reduction
    ...

def gin_meld_reward(completions, **kwargs):
    """Reward for holding/completing melds."""
    # Count melds in hand after action
    # +0.3 per complete meld in hand
    ...
```

**Key differences from Liar's Dice:**

| Aspect | Liar's Dice | Gin Rummy |
|---|---|---|
| Action space | ~10 options | ~50+ (which card to discard) |
| Episodes length | Short (5–15 turns) | Medium (20–40 turns) |
| Reward signal | Binary (valid/invalid) | Continuous (deadwood score) |
| Hidden info | Opponent dice | Stock pile, opponent hand |
| Recommended samples | 2,000–5,000 | 5,000–10,000 |

---

## 13. Tips, Gotchas & Next Steps

### Common Problems & Fixes

**Problem:** Model always outputs CHALLENGE regardless of game state.
**Fix:** Add penalty for challenging when bid is likely true. Check strategy_reward logic.

**Problem:** `CUDA out of memory`
**Fix:** Reduce `num_generations` from 8 to 4, reduce `max_completion_length` to 128.

**Problem:** Reward stays flat at 0 after 100+ steps.
**Fix:** Check reward functions are returning non-zero values. Add `print()` statements
inside rewards during first 10 steps.

**Problem:** KL divergence explodes (goes > 5.0).
**Fix:** Increase `kl_coef` to 0.1, reduce `learning_rate` to 1e-6.

**Problem:** Model ignores `<action>` format and just outputs free text.
**Fix:** Increase `format_reward` weight. Consider an initial SFT warmup phase.

### Next Steps (Advanced)

1. **Self-play:** Have two LLM instances play each other. Use win/loss as the reward.
2. **Curriculum learning:** Start with easy scenarios (obvious valid bids), gradually add harder ones.
3. **Multi-turn episodes:** Use TRL's Environment API (shown in Section 6's `IncrementEnv` example) for full game episodes rather than single-turn decisions.
4. **Scale up:** Try Qwen3-1.7B or Qwen3-4B if you have a 24GB GPU.

---

## 14. Reference Videos & Articles

### 🎓 Courses & Interactive Guides
- **DeepLearning.AI — "Reinforcement Fine-Tuning LLMs with GRPO"**
  https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/
  *Uses Wordle (a word game!) as the training example — closest to our use case.*

- **HuggingFace GRPO Cookbook**
  https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl
  *Official step-by-step guide using Qwen2-0.5B + TRL GRPOTrainer.*

- **HuggingFace LLM Course — Chapter 12: GRPO**
  https://huggingface.co/learn/llm-course/en/chapter12/4

### 📖 Deep-Dive Articles
- **"The State of Reinforcement Learning for LLM Reasoning" — Sebastian Raschka**
  https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training
  *Comprehensive overview of GRPO vs PPO with math.*

- **"GRPO Deep Dive" — Cameron Wolfe (Deep Learning Focus)**
  https://cameronrwolfe.substack.com/p/grpo
  *Best theoretical explanation of how GRPO advantage estimation works.*

- **"PPO & GRPO Guide" — Yuge (Jimmy) Shi**
  https://yugeten.github.io/posts/2025/01/ppogrpo/
  *Visual, intuitive walkthrough of the math.*

### 🔧 Unsloth (Efficient GRPO on Consumer GPUs)
- **Unsloth RL Guide** (covers Qwen3 specifically!)
  https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide
  *5GB VRAM minimum with Unsloth. Includes Qwen3 examples.*

### 🎮 Game-Specific RL Papers & Repos
- **GRL: Game Reinforcement Learning Framework**
  https://github.com/lmgame-org/GRL
  *Multi-turn RL training of LLMs on actual games.*

- **simple_GRPO (200-line minimal implementation)**
  https://github.com/lsdefine/simple_GRPO
  *Great for understanding the GRPO training loop from scratch.*

- **verl: Volcano Engine RL for LLMs**
  https://github.com/verl-project/verl
  *Production-grade GRPO with Qwen3 support.*

---

*Happy training! The key insight: games give you free, verifiable reward functions.
Start with format + valid_move rewards to get stable training, then add strategy rewards.*
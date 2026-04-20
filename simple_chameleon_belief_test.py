#!/usr/bin/env python3
"""Comprehensive test comparing belief updates with vs without HuggingFace hidden states."""

import os
import sys
from typing import Dict, List, Tuple
import torch
import time

# Ensure the package path points to the local chatarena package
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "chatarena"))

from chatarena.backends.llm import TransformersHuggingFaceChat
from chatarena.environments.chameleon_grpo import Chameleon


def compute_belief_entropy(beliefs: torch.Tensor) -> float:
    """Compute Shannon entropy of belief distribution."""
    probs = beliefs + 1e-9
    return float(-(probs * torch.log(probs)).sum())


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute KL(p||q) divergence between two distributions."""
    p = p + 1e-9
    q = q + 1e-9
    return float((p * (torch.log(p) - torch.log(q))).sum())


def identify_chameleon(env: Chameleon) -> str:
    """Identify who the non-chameleon players think is the chameleon."""
    non_chameleon_players = [p for p in env.players if p.hidden_role == "non_chameleon"]
    if not non_chameleon_players:
        return "UNKNOWN"
    
    # Average belief across non-chameleon players
    avg_belief = torch.zeros(len(env.player_names))
    for player in non_chameleon_players:
        avg_belief += player.beliefs
    avg_belief /= len(non_chameleon_players)
    
    chameleon_idx = torch.argmax(avg_belief).item()
    return env.player_names[chameleon_idx]


def run_game(use_hidden_states: bool, game_idx: int, num_clues: int = 6) -> Dict:
    """Run a single game with given configuration."""
    backend = TransformersHuggingFaceChat(
        model="distilgpt2",
        device=-1,
        torch_dtype="float32",
        max_new_tokens=16,
        temperature=0.7,
        do_sample=False,
        use_hidden_states=use_hidden_states,
    )

    players = [
        {"name": "Alice", "role_desc": "You are a clue giver."},
        {"name": "Bob", "role_desc": "You are a clue giver."},
        {"name": "Charlie", "role_desc": "You are a clue giver."},
    ]

    env = Chameleon(
        player_configs=players,
        backend=backend,
        embedding_size=384,
        belief_state_size=128,
        speaker_embedding_size=16,
        num_clue_rounds=1,
    )

    env.reset()
    
    test_clues = [
        "It is a short yellow fruit.",
        "Monkeys love eating this.",
        "A potassium-rich snack.",
        "Often eaten in pasta dishes.",
        "Contains lots of carbs.",
        "Grown in Italy.",
    ]

    game_stats = {
        "game_idx": game_idx,
        "topic": env.topic,
        "secret_word": env.code,
        "actual_chameleon": env.chameleon_name,
        "clues": [],
        "clue_rewards": [],
        "belief_entropies_over_time": {p.name: [compute_belief_entropy(p.beliefs)] for p in env.players},
        "chameleon_word_prob_over_time": [],
        "initial_entropies": {p.name: compute_belief_entropy(p.beliefs) for p in env.players},
        "final_entropies": {},
        "correct_chameleon_identification": False,
        "confidence_in_chameleon": 0.0,
    }

    # Track initial chameleon word probability
    for player in env.players:
        if player.hidden_role == "chameleon":
            word_idx = env.word_to_idx[env.code]
            game_stats["chameleon_word_prob_over_time"].append(float(player.beliefs[word_idx]))

    for clue_idx, clue in enumerate(test_clues[:num_clues]):
        speaker_idx = clue_idx % len(env.player_names)
        speaker_name = env.player_names[speaker_idx]

        # Update beliefs
        env._update_beliefs_for_new_clue(speaker_name=speaker_name, action=clue)
        reward = env.evaluate_clue(speaker_name=speaker_name, action=clue)

        game_stats["clues"].append({
            "clue_text": clue,
            "speaker": speaker_name,
            "reward": float(reward),
        })
        game_stats["clue_rewards"].append(float(reward))

        # Track belief changes
        for player in env.players:
            entropy = compute_belief_entropy(player.beliefs)
            game_stats["belief_entropies_over_time"][player.name].append(entropy)
            
            if player.hidden_role == "chameleon":
                word_idx = env.word_to_idx[env.code]
                game_stats["chameleon_word_prob_over_time"].append(float(player.beliefs[word_idx]))

    # Final statistics
    for player in env.players:
        game_stats["final_entropies"][player.name] = compute_belief_entropy(player.beliefs)

    # Chameleon identification accuracy
    identified_chameleon = identify_chameleon(env)
    game_stats["correct_chameleon_identification"] = (identified_chameleon == env.chameleon_name)
    
    # Confidence in chameleon identification (how much higher was the suspected chameleon)
    non_chameleon_players = [p for p in env.players if p.hidden_role == "non_chameleon"]
    if non_chameleon_players:
        avg_belief = torch.zeros(len(env.player_names))
        for player in non_chameleon_players:
            avg_belief += player.beliefs
        avg_belief /= len(non_chameleon_players)
        
        chameleon_prob = float(avg_belief[env.player_names.index(env.chameleon_name)])
        other_probs = [float(avg_belief[i]) for i in range(len(env.player_names)) if env.player_names[i] != env.chameleon_name]
        avg_other = sum(other_probs) / len(other_probs) if other_probs else 0.5
        game_stats["confidence_in_chameleon"] = chameleon_prob - avg_other

    return game_stats


def run_full_test_suite(use_hidden_states: bool, num_games: int = 3, num_clues: int = 6) -> Dict:
    """Run multiple games and aggregate statistics."""
    mode_label = "Hidden States" if use_hidden_states else "Sentence Encoder"
    print(f"\n{'='*80}")
    print(f"RUNNING TEST SUITE: {mode_label} (use_hidden_states={use_hidden_states})")
    print(f"Games: {num_games}, Clues per game: {num_clues}")
    print(f"{'='*80}")

    all_games = []
    for game_idx in range(num_games):
        print(f"\n  Game {game_idx + 1}/{num_games}...", end=" ", flush=True)
        start = time.time()
        game_stats = run_game(use_hidden_states, game_idx, num_clues)
        elapsed = time.time() - start
        all_games.append(game_stats)
        print(f"({elapsed:.1f}s)")

    # Aggregate statistics
    aggregated = {
        "mode": mode_label,
        "num_games": num_games,
        "num_clues": num_clues,
        "games": all_games,
        "mean_clue_reward": sum(sum(g["clue_rewards"]) for g in all_games) / (num_games * num_clues),
        "total_clue_reward": sum(sum(g["clue_rewards"]) for g in all_games),
        "chameleon_identification_accuracy": sum(g["correct_chameleon_identification"] for g in all_games) / num_games,
        "mean_chameleon_confidence": sum(g["confidence_in_chameleon"] for g in all_games) / num_games,
        "mean_chameleon_word_prob_final": sum(g["chameleon_word_prob_over_time"][-1] for g in all_games) / num_games,
        "final_entropy_changes": {},
    }

    # Average entropy changes per player
    for player_name in all_games[0]["initial_entropies"].keys():
        initial_ents = [g["initial_entropies"][player_name] for g in all_games]
        final_ents = [g["final_entropies"][player_name] for g in all_games]
        mean_initial = sum(initial_ents) / len(initial_ents)
        mean_final = sum(final_ents) / len(final_ents)
        aggregated["final_entropy_changes"][player_name] = {
            "initial": mean_initial,
            "final": mean_final,
            "delta": mean_initial - mean_final,
        }

    return aggregated


def print_test_results(results: Dict) -> None:
    """Pretty-print test results."""
    print(f"\n{'─'*80}")
    print(f"Results: {results['mode']}")
    print(f"{'─'*80}")
    print(f"Games: {results['num_games']} × {results['num_clues']} clues")
    print(f"\nReward Metrics:")
    print(f"  Mean reward per clue:   {results['mean_clue_reward']:+.6f}")
    print(f"  Total reward (all clues): {results['total_clue_reward']:+.6f}")
    print(f"\nChameleon Detection:")
    print(f"  Identification accuracy:  {results['chameleon_identification_accuracy']:.1%}")
    print(f"  Mean confidence score:    {results['mean_chameleon_confidence']:+.6f}")
    print(f"  Mean prob on secret word: {results['mean_chameleon_word_prob_final']:.6f}")
    print(f"\nBelief Confidence (Entropy):")
    for player_name, changes in results["final_entropy_changes"].items():
        delta = changes["delta"]
        direction = "↓" if delta > 0 else "↑"
        print(f"  {player_name:10s}: {changes['initial']:.4f} → {changes['final']:.4f} (Δ={delta:+.4f}) {direction}")


def compare_results(results_hidden: Dict, results_encoder: Dict) -> None:
    """Compare results between two modes."""
    print(f"\n{'='*80}")
    print("DETAILED COMPARISON: Hidden States vs Sentence Encoder")
    print(f"{'='*80}")

    print(f"\n{'Metric':<40} {'Hidden States':>18} {'Encoder':>18} {'Difference':>15}")
    print("─" * 95)

    metrics = [
        ("Mean reward per clue", "mean_clue_reward"),
        ("Total reward", "total_clue_reward"),
        ("Chameleon ID accuracy", "chameleon_identification_accuracy"),
        ("Chameleon confidence", "mean_chameleon_confidence"),
        ("Secret word prob (chameleon)", "mean_chameleon_word_prob_final"),
    ]

    for display_name, key in metrics:
        val_hidden = results_hidden[key]
        val_encoder = results_encoder[key]
        
        # Format appropriately
        if "accuracy" in display_name:
            fmt_h = f"{val_hidden:.1%}"
            fmt_e = f"{val_encoder:.1%}"
            diff = val_hidden - val_encoder
            fmt_d = f"{diff:+.1%}"
        else:
            fmt_h = f"{val_hidden:+.6f}"
            fmt_e = f"{val_encoder:+.6f}"
            diff = val_hidden - val_encoder
            fmt_d = f"{diff:+.6f}"

        winner = "✓ Hidden" if diff > 0 else "✓ Encoder" if diff < 0 else "  EQUAL"
        print(f"{display_name:<40} {fmt_h:>18} {fmt_e:>18} {fmt_d:>15} {winner}")

    print(f"\n{'Entropy Changes:':<40}")
    print("─" * 95)
    print(f"{'Player':<20} {'Hidden States':>25} {'Sentence Encoder':>25} {'Difference':>20}")
    print("─" * 95)
    for player_name in results_hidden["final_entropy_changes"]:
        h_delta = results_hidden["final_entropy_changes"][player_name]["delta"]
        e_delta = results_encoder["final_entropy_changes"][player_name]["delta"]
        diff = h_delta - e_delta
        winner = "✓ Hidden" if diff > 0 else "✓ Encoder" if diff < 0 else "  EQUAL"
        print(f"{player_name:<20} {h_delta:>15.6f} (Δ)  {e_delta:>15.6f} (Δ)  {diff:>15.6f} {winner}")


def main() -> None:
    print("\n" + "="*80)
    print("COMPREHENSIVE CHAMELEON BELIEF TEST")
    print("Hidden States vs Sentence Encoder Comparison")
    print("="*80)

    # Run both test suites
    num_games = 3
    num_clues = 6

    results_hidden = run_full_test_suite(use_hidden_states=True, num_games=num_games, num_clues=num_clues)
    results_encoder = run_full_test_suite(use_hidden_states=False, num_games=num_games, num_clues=num_clues)

    # Print individual results
    print_test_results(results_hidden)
    print_test_results(results_encoder)

    # Print comparison
    compare_results(results_hidden, results_encoder)

    print(f"\n{'='*80}")
    print("Test complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

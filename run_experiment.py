#!/usr/bin/env python3
"""Unified entry point for running Chameleon experiments."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Chameleon game experiments with closed- or open-source LLMs.",
    )
    parser.add_argument(
        "mode",
        choices=["cs", "cs-belief", "os", "grpo"],
        help="Experiment mode: 'cs' (closed-source, no beliefs), 'cs-belief' (closed-source with belief voting), 'os' (open-source baseline), or 'grpo' (open-source with GRPO training).",
    )
    parser.add_argument(
        "--config",
        default="chatarena/examples/chameleon_closed_3p.json",
        help="Path to the arena config file.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Name for this experiment (used in log/summary filenames).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per game.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.9 for cs, 0.7 for os).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for logs and summaries.",
    )
    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Save full conversation transcripts.",
    )

    # Closed-source options
    cs_group = parser.add_argument_group("closed-source options")
    cs_group.add_argument(
        "--backend",
        choices=["openai", "claude", "gemini"],
        default="openai",
        help="Which API backend to use.",
    )
    cs_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per response.",
    )

    # Open-source options
    os_group = parser.add_argument_group("open-source options")
    os_group.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or path.",
    )
    os_group.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (-1 for CPU).",
    )
    os_group.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )
    os_group.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Max new tokens to generate per turn.",
    )
    os_group.add_argument(
        "--eval-only",
        action="store_true",
        help="GRPO mode only: skip policy updates and just play games with the current weights.",
    )
    os_group.add_argument(
        "--eval-runs",
        type=int,
        default=0,
        help="GRPO mode only: after training num_runs games, play this many additional games "
             "with frozen LoRA weights for evaluation. Logs go to <experiment-id>_eval_*.",
    )
    os_group.add_argument(
        "--clue-number",
        type=int,
        default=8,
        help="GRPO mode only: number of candidate clues each non-chameleon generates per turn.",
    )

    reward_group = parser.add_argument_group("grpo reward weights")
    reward_group.add_argument("--reward-alpha", type=float, default=0.5,
        help="Self-suspicion coefficient (default: 0.5).")
    reward_group.add_argument("--reward-gamma", type=float, default=2.0,
        help="Word-leak coefficient (default: 2.0).")
    reward_group.add_argument("--reward-word-leak-threshold", type=float, default=0.15,
        help="Word-leak threshold below which leak is ignored (default: 0.15).")
    reward_group.add_argument("--reward-max-tokens", type=int, default=12,
        help="Token count above which length penalty starts (default: 12).")
    reward_group.add_argument("--reward-zeta", type=float, default=0.1,
        help="Length-penalty rate; penalty=exp(zeta*over_by)-1 (default: 0.1).")
    reward_group.add_argument("--reward-length-cap", type=float, default=2.0,
        help="Max magnitude of length penalty; penalty=min(exp(zeta*over)-1, cap) (default: 2.0).")

    train_group = parser.add_argument_group("grpo training hyperparameters")
    train_group.add_argument("--policy-lr", type=float, default=2e-5,
        help="Adam learning rate for policy (LoRA) updates (default: 2e-5).")
    train_group.add_argument("--grpo-beta", type=float, default=0.3,
        help="KL penalty coefficient against the reference (LoRA-disabled) policy (default: 0.3).")

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = f"chameleon-{args.mode}"

    if args.mode == "cs":
        from chatarena.experiments.cs_experiment import ClosedSourceExperiment

        kwargs = dict(
            experiment_filepath=args.config,
            backend_name=args.backend,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            save_transcript=args.save_transcript,
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens

        exp = ClosedSourceExperiment(**kwargs)

    elif args.mode == "cs-belief":
        from chatarena.experiments.cs_belief_experiment import CSBeliefExperiment

        kwargs = dict(
            experiment_filepath=args.config,
            backend_name=args.backend,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            save_transcript=args.save_transcript,
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens

        exp = CSBeliefExperiment(**kwargs)

    elif args.mode == "os":
        from chatarena.experiments.os_experiment import OpenSourceExperiment

        kwargs = dict(
            experiment_filepath=args.config,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            save_transcript=args.save_transcript,
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature

        exp = OpenSourceExperiment(**kwargs)

    else:  # grpo
        from chatarena.experiments.grpo_experiment import GRPOExperiment

        kwargs = dict(
            experiment_filepath=args.config,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            max_steps=args.max_steps,
            log_dir=args.log_dir,
            save_transcript=args.save_transcript,
            eval_only=args.eval_only,
            eval_num_runs=args.eval_runs,
            clue_number=args.clue_number,
            reward_alpha=args.reward_alpha,
            reward_gamma=args.reward_gamma,
            reward_word_leak_threshold=args.reward_word_leak_threshold,
            reward_max_tokens=args.reward_max_tokens,
            reward_zeta=args.reward_zeta,
            reward_length_cap=args.reward_length_cap,
            policy_lr=args.policy_lr,
            grpo_beta=args.grpo_beta,
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature

        exp = GRPOExperiment(**kwargs)

    exp.run()


if __name__ == "__main__":
    main()

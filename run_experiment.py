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
        default=128,
        help="Max new tokens to generate per turn.",
    )

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
        )
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature

        exp = GRPOExperiment(**kwargs)

    exp.run()


if __name__ == "__main__":
    main()

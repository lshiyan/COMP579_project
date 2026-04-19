import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..chatarena.agent import Player
from ..chatarena.arena import Arena
from ..chatarena.backends import load_backend
from ..chatarena.config import ArenaConfig, BackendConfig
from ..chatarena.environments import load_environment


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BACKEND_CONFIG = {
    "backend_type": "transformers:huggingface-chat",
    "model": DEFAULT_MODEL,
    "device": 0,
    "torch_dtype": "bfloat16",
    "max_new_tokens": 32,
    "temperature": 0.7,
    "do_sample": True,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an open-source LLM experiment in ChatArena.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        required=True,
        metavar="FILE",
        help="Path to the arena config file (JSON).",
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Name of experiment",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or local path.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (-1 for CPU).",
    )

    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate per turn.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of experiment runs.",
    )

    parser.add_argument(
        "--log-dir",
        default="logs",
        metavar="DIR",
        help="Directory where experiment logs are written.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity of console output.",
    )

    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Write the full conversation transcript to --log-dir.",
    )

    return parser


def setup_logging(log_dir: str, experiment_id: str | None, level: str) -> logging.Logger:
    logger = logging.getLogger(f"open_source_experiment.{experiment_id or 'run'}")
    logger.setLevel(getattr(logging, level))
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    stem = experiment_id or "run"
    log_path = os.path.join(log_dir, f"{stem}.log")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class OpenSourceExperiment:
    def __init__(
        self,
        experiment_filepath: str,
        model: str = DEFAULT_MODEL,
        device: int = 0,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        experiment_id: str | None = None,
        num_runs: int = 1,
        max_steps: int | None = 50,
        log_dir: str = "logs",
        log_level: str = "INFO",
        save_transcript: bool = False,
    ):
        if num_runs < 1:
            raise ValueError("num_runs must be at least 1")

        self.experiment_filepath = experiment_filepath
        self.model_name = model
        self.experiment_id = experiment_id
        self.num_runs = num_runs
        self.max_steps = max_steps
        self.save_transcript = save_transcript
        self.log_dir = log_dir

        self.logger = setup_logging(log_dir, experiment_id, log_level)
        self.logger.info("Loading config from %s", experiment_filepath)

        self.arena_config = ArenaConfig.load(experiment_filepath)

        backend_cfg = dict(BACKEND_CONFIG)
        backend_cfg["model"] = model
        backend_cfg["device"] = device
        backend_cfg["torch_dtype"] = torch_dtype
        backend_cfg["max_new_tokens"] = max_new_tokens
        backend_cfg["temperature"] = temperature

        self.logger.info("Loading model %s on device %d ...", model, device)
        self.shared_backend = load_backend(BackendConfig(backend_cfg))
        self.logger.info("Model loaded.")

        self.logger.info(
            "Experiment ready | model=%s | device=%d | players=%d | runs=%d",
            model,
            device,
            len(self.arena_config.players),
            self.num_runs,
        )

    def _build_arena(self) -> Arena:
        """Build a fresh Arena where all players share the pre-loaded backend."""
        global_prompt = self.arena_config.get("global_prompt", None)

        players = []
        for player_config in self.arena_config.players:
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt
            player = Player(
                name=player_config["name"],
                role_desc=player_config["role_desc"],
                backend=self.shared_backend,
                global_prompt=player_config.get("global_prompt"),
            )
            players.append(player)

        player_names = [p.name for p in players]
        self.arena_config.environment["player_names"] = player_names
        env = load_environment(self.arena_config.environment)

        return Arena(players, env, global_prompt=global_prompt)

    def _extract_run_result(self, final_timestep: Any, run_idx: int) -> dict[str, Any]:
        return {
            "run_idx": run_idx,
            "chameleon_won": None if final_timestep is None else getattr(final_timestep, "chameleon_won", None),
            "win_method": None if final_timestep is None else getattr(final_timestep, "win_method", None),
        }

    def run_once(self, run_idx: int) -> dict[str, Any]:
        self.logger.info(
            "Starting run %d/%d (max_steps=%s)",
            run_idx,
            self.num_runs,
            self.max_steps,
        )

        arena = self._build_arena()
        final_timestep = None
        t0 = time.monotonic()

        try:
            final_timestep = arena.run(num_steps=self.max_steps)
        except KeyboardInterrupt:
            self.logger.warning("Run %d interrupted by user.", run_idx)
            raise
        finally:
            if self.save_transcript:
                self._save_transcript(arena, run_idx)

        elapsed = time.monotonic() - t0
        result = self._extract_run_result(final_timestep, run_idx)
        result["elapsed_s"] = round(elapsed, 2)
        self.logger.info(
            "Finished run %d/%d | chameleon_won=%s | win_method=%s | %.1fs",
            run_idx,
            self.num_runs,
            result["chameleon_won"],
            result["win_method"],
            elapsed,
        )
        return result

    def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for run_idx in range(1, self.num_runs + 1):
            results.append(self.run_once(run_idx))

        self._log_summary(results)
        self._save_summary(results)
        return results

    def _save_transcript(self, arena: Arena, run_idx: int) -> None:
        stem = self.experiment_id or "run"
        path = os.path.join(self.log_dir, f"{stem}_transcript.txt")

        messages = getattr(arena, "messages", None)
        if messages is None:
            env = getattr(arena, "environment", None)
            pool = getattr(env, "message_pool", None)
            messages = getattr(pool, "get_all_messages", lambda: [])()

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  GAME {run_idx}/{self.num_runs}\n")
            f.write(f"{'=' * 60}\n\n")
            for msg in messages:
                f.write(f"{msg}\n")

        self.logger.info("Transcript saved → %s", path)

    def _log_summary(self, results: list[dict[str, Any]]) -> None:
        total = len(results)
        chameleon_wins = sum(r["chameleon_won"] is True for r in results)
        non_chameleon_wins = sum(r["chameleon_won"] is False for r in results)
        unknown = sum(r["chameleon_won"] is None for r in results)

        self.logger.info(
            "Summary | total_runs=%d | chameleon_wins=%d | non_chameleon_wins=%d | unknown=%d",
            total,
            chameleon_wins,
            non_chameleon_wins,
            unknown,
        )

    def _save_summary(self, results: list[dict[str, Any]]) -> None:
        stem = self.experiment_id or "run"
        path = os.path.join(self.log_dir, f"{stem}_summary.txt")

        total = len(results)
        chameleon_wins = sum(r["chameleon_won"] is True for r in results)
        non_chameleon_wins = sum(r["chameleon_won"] is False for r in results)
        unknown = sum(r["chameleon_won"] is None for r in results)

        win_method_counts = Counter(r.get("win_method") or "unknown" for r in results)
        total_time = sum(r.get("elapsed_s", 0) for r in results)
        avg_time = total_time / total if total else 0

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"experiment_id: {stem}\n")
            f.write(f"model: {self.model_name}\n")
            f.write(f"total_runs: {total}\n")
            f.write(f"chameleon_wins: {chameleon_wins}\n")
            f.write(f"non_chameleon_wins: {non_chameleon_wins}\n")
            f.write(f"unknown: {unknown}\n")
            f.write(f"total_time_s: {total_time:.1f}\n")
            f.write(f"avg_time_per_game_s: {avg_time:.1f}\n")

            f.write("\nwin_method_summary:\n")
            for method, count in sorted(win_method_counts.items()):
                f.write(f"  {method}: {count}\n")

            f.write("\nper_run_results:\n")
            for r in results:
                f.write(
                    f"  run={r['run_idx']}, "
                    f"chameleon_won={r.get('chameleon_won')}, "
                    f"win_method={r.get('win_method')}, "
                    f"elapsed_s={r.get('elapsed_s', 0)}\n"
                )

        self.logger.info("Summary saved → %s", path)

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "OpenSourceExperiment":
        parser = build_parser()
        args = parser.parse_args(argv)
        return cls(
            experiment_filepath=args.experiment,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            log_dir=args.log_dir,
            log_level=args.log_level,
            save_transcript=args.save_transcript,
        )


if __name__ == "__main__":
    exp = OpenSourceExperiment.from_args()
    exp.run()

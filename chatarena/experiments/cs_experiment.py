import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any
from collections import Counter

from ..chatarena.arena import Arena
from ..chatarena.config import ArenaConfig, BackendConfig

BACKEND_CONFIGS = {
    "openai": {
        "backend_type": "openai",
        "temperature": 0.9,
        "max_tokens": 100,
    },
    "claude": {
        "backend_type": "claude",
        "temperature": 0.9,
        "max_tokens": 100,
    },
    "gemini": {
        "backend_type": "gemini",
        "temperature": 0.9,
        "max_output_tokens": 100,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a closed-source LLM experiment in ChatArena.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        required=True,
        metavar="FILE",
        help="Path to the arena config file (JSON/YAML).",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Name of experiment",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=list(BACKEND_CONFIGS.keys()),
        help="Which closed-source backend to assign to every player.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        required=True,
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
    logger = logging.getLogger(f"closed_source_experiment.{experiment_id or 'run'}")
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


class ClosedSourceExperiment:
    def __init__(
        self,
        experiment_filepath: str,
        backend_name: str,
        experiment_id: str | None = None,
        num_runs: int = 1,
        max_steps: int | None = 50,
        temperature: float | None = None,
        max_tokens: int | None = None,
        log_dir: str = "logs",
        log_level: str = "INFO",
        save_transcript: bool = False,
    ):
        if num_runs < 1:
            raise ValueError("num_runs must be at least 1")

        self.experiment_filepath = experiment_filepath
        self.backend_name = backend_name
        self.experiment_id = experiment_id
        self.num_runs = num_runs
        self.max_steps = max_steps
        self.save_transcript = save_transcript
        self.log_dir = log_dir
        self.log_level = log_level
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.logger = setup_logging(log_dir, experiment_id, log_level)
        self.logger.info("Loading config from %s", experiment_filepath)

        self.arena_config = ArenaConfig.load(experiment_filepath)

        self.backend_cfg = dict(BACKEND_CONFIGS[backend_name])
        if temperature is not None:
            self.backend_cfg["temperature"] = temperature
            self.logger.debug("Temperature overridden → %.2f", temperature)
        if max_tokens is not None:
            token_key = "max_output_tokens" if backend_name == "gemini" else "max_tokens"
            self.backend_cfg[token_key] = max_tokens
            self.logger.debug("Token limit overridden → %d", max_tokens)

        self._apply_backend_config()
        self.logger.info(
            "Experiment ready | backend=%s | players=%d | runs=%d",
            backend_name,
            len(self.arena_config.players),
            self.num_runs,
        )

    def _apply_backend_config(self) -> None:
        for player in self.arena_config.players:
            player.backend = BackendConfig(dict(self.backend_cfg))

    def _build_arena(self) -> Arena:
        self._apply_backend_config()
        return Arena.from_config(self.arena_config)

    def _extract_run_result(self, final_timestep: Any, run_idx: int) -> dict[str, Any]:
        return {
            "run_idx": run_idx,
            "chameleon_won": None if final_timestep is None else getattr(final_timestep, "chameleon_won", None),
            "win_method": None if final_timestep is None else getattr(final_timestep, "win_method", None),
        }

    def run_once(self, run_idx: int) -> dict[str, Any]:
        run_name = f"{self.experiment_id or 'run'}_run{run_idx:03d}"
        self.logger.info(
            "Starting run %d/%d (max_steps=%s)",
            run_idx,
            self.num_runs,
            self.max_steps,
        )

        arena = self._build_arena()
        final_timestep = None

        try:
            final_timestep = arena.run(num_steps=self.max_steps)
        except KeyboardInterrupt:
            self.logger.warning("Run %d interrupted by user.", run_idx)
            raise
        finally:
            if self.save_transcript:
                self._save_transcript(arena, run_name)

        result = self._extract_run_result(final_timestep, run_idx)
        self.logger.info(
            "Finished run %d/%d | chameleon_won=%s | win_method=%s",
            run_idx,
            self.num_runs,
            result["chameleon_won"],
            result["win_method"]
        )
        return result

    def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for run_idx in range(1, self.num_runs + 1):
            results.append(self.run_once(run_idx))

        self._log_summary(results)
        self._save_summary(results)
        return results

    def _save_transcript(self, arena: Arena, stem: str) -> None:
        path = os.path.join(self.log_dir, f"{stem}_transcript.txt")

        messages = getattr(arena, "messages", None)
        if messages is None:
            env = getattr(arena, "environment", None)
            pool = getattr(env, "message_pool", None)
            messages = getattr(pool, "get_all_messages", lambda: [])()

        with open(path, "w", encoding="utf-8") as f:
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

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"experiment_id: {stem}\n")
            f.write(f"backend: {self.backend_name}\n")
            f.write(f"total_runs: {total}\n")
            f.write(f"chameleon_wins: {chameleon_wins}\n")
            f.write(f"non_chameleon_wins: {non_chameleon_wins}\n")
            f.write(f"unknown: {unknown}\n")

            f.write("\nwin_method_summary:\n")
            for method, count in sorted(win_method_counts.items()):
                f.write(f"  {method}: {count}\n")

            f.write("\nper_run_results:\n")
            for r in results:
                f.write(
                    f"  run={r['run_idx']}, "
                    f"chameleon_won={r.get('chameleon_won')}, "
                    f"win_method={r.get('win_method')}\n"
                )

        self.logger.info("Summary saved → %s", path)

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "ClosedSourceExperiment":
        parser = build_parser()
        args = parser.parse_args(argv)
        return cls(
            experiment_filepath=args.experiment,
            backend_name=args.backend,
            experiment_id=args.experiment_id,
            num_runs=args.num_runs,
            log_dir=args.log_dir,
            log_level=args.log_level,
            save_transcript=args.save_transcript,
        )


if __name__ == "__main__":
    exp = ClosedSourceExperiment.from_args()
    exp.run()
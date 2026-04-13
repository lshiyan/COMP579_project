import argparse
import logging
import os
import sys
from pathlib import Path

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

    # --- required ---
    parser.add_argument(
        "--experiment",
        required=True,
        metavar="FILE",
        help="Path to the arena config file (JSON/YAML).",
    )
    
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Name of experiment"
    )
    
    parser.add_argument(
        "--backend",
        required=True,
        choices=list(BACKEND_CONFIGS.keys()),
        help="Which closed-source backend to assign to every player.",
    )

    # --- logging / output ---
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
    logger = logging.getLogger("closed_source_experiment")
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
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
        max_steps: int | None = 50,
        temperature: float | None = None,
        max_tokens: int | None = None,
        log_dir: str = "logs",
        log_level: str = "INFO",
        save_transcript: bool = False,
    ):
        self.experiment_filepath = experiment_filepath
        self.experiment_id = experiment_id
        self.max_steps = max_steps
        self.save_transcript = save_transcript
        self.log_dir = log_dir

        self.logger = setup_logging(log_dir, experiment_id, log_level)
        self.logger.info("Loading config from %s", experiment_filepath)

        self.arena_config = ArenaConfig.load(experiment_filepath)

        # Build backend config, applying any CLI overrides
        backend_cfg = dict(BACKEND_CONFIGS[backend_name])
        if temperature is not None:
            backend_cfg["temperature"] = temperature
            self.logger.debug("Temperature overridden → %.2f", temperature)
        if max_tokens is not None:
            token_key = "max_output_tokens" if backend_name == "gemini" else "max_tokens"
            backend_cfg[token_key] = max_tokens
            self.logger.debug("Token limit overridden → %d", max_tokens)

        for player in self.arena_config.players:
            player.backend = BackendConfig(backend_cfg)

        self.arena = Arena.from_config(self.arena_config)
        self.logger.info(
            "Arena ready | backend=%s | players=%d",
            backend_name,
            len(self.arena_config.players),
        )

    def run(self) -> None:
        self.logger.info("Starting experiment (max_steps=%s)", self.max_steps)
        try:
            final_timestep = self.arena.run(num_steps=self.max_steps)
        except KeyboardInterrupt:
            self.logger.warning("Run interrupted by user.")
        finally:
            if self.save_transcript:
                self._save_transcript()

    def _save_transcript(self) -> None:
        stem = self.experiment_id or "run"
        path = os.path.join(self.log_dir, f"{stem}_transcript.txt")
        # arena.messages is a common ChatArena convention; adjust if needed
        messages = getattr(self.arena, "messages", [])
        with open(path, "w") as f:
            for msg in messages:
                f.write(f"{msg}\n")
        self.logger.info("Transcript saved → %s", path)

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> "ClosedSourceExperiment":
        parser = build_parser()
        args = parser.parse_args(argv)
        return cls(
            experiment_filepath=args.experiment,
            backend_name=args.backend,
            experiment_id=args.experiment_id,
            log_dir=args.log_dir,
            log_level=args.log_level,
            save_transcript=args.save_transcript,
        )


if __name__ == "__main__":
    exp = ClosedSourceExperiment.from_args()
    exp.run()
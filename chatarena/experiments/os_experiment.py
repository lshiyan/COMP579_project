import argparse
import logging
import os
import sys
from pathlib import Path

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
    "max_new_tokens": 128,
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
        default=128,
        help="Maximum new tokens to generate per turn.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
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
    logger = logging.getLogger("open_source_experiment")
    logger.setLevel(level)

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
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        experiment_id: str | None = None,
        max_steps: int | None = 50,
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

        backend_cfg = dict(BACKEND_CONFIG)
        backend_cfg["model"] = model
        backend_cfg["device"] = device
        backend_cfg["torch_dtype"] = torch_dtype
        backend_cfg["max_new_tokens"] = max_new_tokens
        backend_cfg["temperature"] = temperature

        # Load the model once and share across all players
        self.logger.info("Loading model %s on device %d ...", model, device)
        shared_backend = load_backend(BackendConfig(backend_cfg))
        self.logger.info("Model loaded.")

        self.arena = self._build_arena(self.arena_config, shared_backend)
        self.logger.info(
            "Arena ready | model=%s | device=%d | players=%d",
            model,
            device,
            len(self.arena_config.players),
        )

    @staticmethod
    def _build_arena(config: ArenaConfig, shared_backend) -> Arena:
        """Build an Arena where all players share a single backend instance."""
        global_prompt = config.get("global_prompt", None)

        players = []
        for player_config in config.players:
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt
            player = Player(
                name=player_config["name"],
                role_desc=player_config["role_desc"],
                backend=shared_backend,
                global_prompt=player_config.get("global_prompt"),
            )
            players.append(player)

        player_names = [p.name for p in players]
        config.environment["player_names"] = player_names
        env = load_environment(config.environment)

        return Arena(players, env, global_prompt=global_prompt)

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
        messages = getattr(self.arena, "messages", [])
        with open(path, "w") as f:
            for msg in messages:
                f.write(f"{msg}\n")
        self.logger.info("Transcript saved → %s", path)

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
            log_dir=args.log_dir,
            log_level=args.log_level,
            save_transcript=args.save_transcript,
        )


if __name__ == "__main__":
    exp = OpenSourceExperiment.from_args()
    exp.run()

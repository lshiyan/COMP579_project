import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..chatarena.backends import load_backend
from ..chatarena.backends.llm import TransformersHuggingFaceChat
from ..chatarena.chameleon_arena import ChameleonArena, RunLogger
from ..chatarena.config import ArenaConfig, BackendConfig
from ..chatarena.environments.chameleon_grpo import Chameleon


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

BACKEND_CONFIG = {
    "backend_type": "transformers:huggingface-chat",
    "model": DEFAULT_MODEL,
    "device": 0,
    "torch_dtype": "bfloat16",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "do_sample": True,
    "lora_cfg": {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
}


def setup_logging(log_dir: str, experiment_id: str | None, level: str) -> logging.Logger:
    logger = logging.getLogger(f"grpo_experiment.{experiment_id or 'run'}")
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


class GRPOExperiment:
    def __init__(
        self,
        experiment_filepath: str,
        model: str = DEFAULT_MODEL,
        device: int = 0,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
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
        assert isinstance(self.shared_backend, TransformersHuggingFaceChat)
        self.logger.info("Model loaded.")

        # Extract player configs from the arena config
        global_prompt = self.arena_config.get("global_prompt", None)
        self.player_configs = []
        for player_config in self.arena_config.players:
            pc = dict(player_config)
            if global_prompt is not None:
                pc["global_prompt"] = global_prompt
            self.player_configs.append(pc)

        self.global_prompt = global_prompt

        self.logger.info(
            "GRPO experiment ready | model=%s | device=%d | players=%d | runs=%d",
            model, device, len(self.player_configs), self.num_runs,
        )

    def _build_arena(self) -> ChameleonArena:
        stem = self.experiment_id or "run"
        run_logger = RunLogger(
            log_dir=self.log_dir,
            log_path=os.path.join(self.log_dir, f"{stem}_arena.log"),
        )

        env = Chameleon(
            player_configs=self.player_configs,
            backend=self.shared_backend,
        )

        return ChameleonArena(
            environment=env,
            global_prompt=self.global_prompt,
            logger=run_logger,
        )

    def _extract_run_result(self, arena: ChameleonArena, run_idx: int) -> dict[str, Any]:
        env = arena.environment
        chameleon_won = None
        win_method = None

        last_msg = env.message_pool.last_message
        if last_msg is not None:
            content = last_msg.content.lower()
            if "won the game" in content or "won!" in content:
                if env.chameleon_name.lower() in content and "won" in content:
                    chameleon_won = True
                    if "guessed the code correctly" in content:
                        win_method = "chameleon-guess"
                    else:
                        win_method = "chameleon-votes"
                else:
                    chameleon_won = False
                    win_method = "non-chameleon"

        return {
            "run_idx": run_idx,
            "chameleon_won": chameleon_won,
            "win_method": win_method,
        }

    def run_once(self, arena: ChameleonArena, run_idx: int) -> dict[str, Any]:
        self.logger.info(
            "Starting run %d/%d (max_steps=%s)",
            run_idx, self.num_runs, self.max_steps,
        )

        arena.reset()
        t0 = time.monotonic()

        try:
            arena.run(num_steps=self.max_steps)
        except KeyboardInterrupt:
            self.logger.warning("Run %d interrupted by user.", run_idx)
            raise

        elapsed = time.monotonic() - t0
        result = self._extract_run_result(arena, run_idx)
        result["elapsed_s"] = round(elapsed, 2)

        if self.save_transcript:
            self._save_transcript(arena, run_idx)

        self.logger.info(
            "Finished run %d/%d | chameleon_won=%s | win_method=%s | %.1fs",
            run_idx, self.num_runs,
            result["chameleon_won"], result["win_method"], elapsed,
        )
        return result

    def run(self) -> list[dict[str, Any]]:
        arena = self._build_arena()
        results: list[dict[str, Any]] = []
        for run_idx in range(1, self.num_runs + 1):
            results.append(self.run_once(arena, run_idx))
        arena.logger.close()
        self._log_summary(results)
        self._save_summary(results)
        return results

    def _save_transcript(self, arena: ChameleonArena, run_idx: int) -> None:
        stem = self.experiment_id or "run"
        path = os.path.join(self.log_dir, f"{stem}_transcript.txt")

        messages = arena.environment.message_pool.get_all_messages()

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
            total, chameleon_wins, non_chameleon_wins, unknown,
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
            f.write(f"mode: grpo\n")
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

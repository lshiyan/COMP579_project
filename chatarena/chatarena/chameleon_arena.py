import atexit
import csv
import json
import logging
import os
import uuid
from datetime import datetime
import torch
from typing import Dict, List, Union

from .agent import Player
from .config import ArenaConfig
from .environments import Environment, TimeStep, load_environment, Chameleon

WIDTH = 80


class TooManyInvalidActions(Exception):
    pass


class RunLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"run_{timestamp}.log")
        self._file = open(self.path, "w")
        self._game_num = 0
        self._write("=" * WIDTH)
        self._write(f"  RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("=" * WIDTH)
        print(f"[Logger] Writing to {self.path}")
        atexit.register(self.close)

    def _write(self, text: str = ""):
        self._file.write(text + "\n")
        self._file.flush()

    def log_game_start(self, topic: str, code: str, chameleon_name: str, player_names: list):
        self._game_num += 1
        self._write()
        self._write("=" * WIDTH)
        self._write(f"  GAME {self._game_num}")
        self._write("=" * WIDTH)
        self._write(f"  Topic: {topic}  |  Secret Word: {code}  |  Chameleon: {chameleon_name}")
        self._write(f"  Players: {', '.join(player_names)}")
        self._write("-" * WIDTH)

    def log_step(
        self,
        player_name: str,
        responses: list,
        belief_rewards: list,
        advantages: list,
        best_idx: int,
        best_action: str,
        grpo_losses: list,
        new_messages: list,
        terminal_rewards: dict,
    ):
        self._write()
        self._write(f"  [ {player_name} ]")

        for i, (resp, reward, adv) in enumerate(zip(responses, belief_rewards, advantages)):
            action_preview = resp["action"].replace("\n", " ").strip()[:120]
            self._write(f"    Attempt {i + 1}  belief_reward={reward:+.4f}  advantage={adv:+.4f}")
            self._write(f"      \"{action_preview}\"")

        best_preview = best_action.replace("\n", " ").strip()[:120]
        self._write(f"    >> Best attempt {best_idx + 1}: \"{best_preview}\"")

        if grpo_losses:
            loss_parts = "  ".join(f"e{i+1}:{l:.4f}" for i, l in enumerate(grpo_losses))
            self._write(f"    GRPO losses: {loss_parts}")

        if new_messages:
            self._write()
            for msg in new_messages:
                visible = msg.visible_to if isinstance(msg.visible_to, str) else ", ".join(msg.visible_to)
                content = msg.content.replace("\n", " ").strip()
                self._write(f"  [{msg.agent_name} -> {visible}]: {content}")

        if terminal_rewards:
            self._write()
            self._write("  --- GAME OVER ---")
            for name, reward in terminal_rewards.items():
                self._write(f"    {name}: {'+' if reward > 0 else ''}{reward:.1f}")

    def close(self):
        if self._file.closed:
            return
        self._write()
        self._write("=" * WIDTH)
        self._write(f"  RUN ENDED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write("=" * WIDTH)
        self._file.close()



class ChameleonArena:
    """Utility class that manages the game environment and players."""

    def __init__(
        self, environment: Chameleon, global_prompt: str = None, clue_number: int = 3, num_grpo_epochs: int = 10
    ):
        # Create a container for the players and environment and reset the game
        self.environment = environment
        self.global_prompt = global_prompt
        self.clue_number = clue_number

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game

        self.reference_model = self.environment.backend.get_ref_model()
        self.num_grpo_epochs = num_grpo_epochs

        self.logger = RunLogger()
        self.logger.log_game_start(
            topic=self.environment.topic,
            code=self.environment.code,
            chameleon_name=self.environment.chameleon_name,
            player_names=self.environment.player_names,
        )
    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.environment.players}

    def reset(self) -> TimeStep:
        self.current_timestep = self.environment.reset()
        self.uuid = uuid.uuid4()
        self.logger.log_game_start(
            topic=self.environment.topic,
            code=self.environment.code,
            chameleon_name=self.environment.chameleon_name,
            player_names=self.environment.player_names,
        )
        return self.current_timestep

    def step(self) -> TimeStep:
        """Take a step in the game: one player takes an action and the environment updates."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]
        observation = self.environment.get_observation(player_name)

        rewards = []
        responses = []
        for _ in range(self.clue_number):
            response = player(observation)
            responses.append(response)
            action = response["action"]
            with torch.no_grad():
                reward = self.environment.evaluate_clue(player_name, action)
                rewards.append(reward)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        grpo_losses = []
        for _ in range(self.num_grpo_epochs):
            loss = self._compute_grpo_loss(player, responses, advantages)
            grpo_losses.append(loss.item())

            #TODO: Setup PEFT optimizer for backend model.
            """optimizer.zero_grad()
            loss.backward()
            optimizer.step()"""

        best_idx = int(advantages.argmax().item())
        best_action = responses[best_idx]["action"]

        msg_count_before = len(self.environment.message_pool._messages)
        timestep = self.environment.step(player_name, best_action)
        new_messages = self.environment.message_pool._messages[msg_count_before:]

        self.logger.log_step(
            player_name=player_name,
            responses=responses,
            belief_rewards=[r.item() for r in rewards],
            advantages=advantages.tolist(),
            best_idx=best_idx,
            best_action=best_action,
            grpo_losses=grpo_losses,
            new_messages=new_messages,
            terminal_rewards=timestep.reward if timestep.terminal else None,
        )

        return timestep

    def _compute_grpo_loss(
        self,
        player: Player,
        responses: list,
        advantages: torch.Tensor,
        eps: float = 0.2,
        beta: float = 0.01,
    ) -> torch.Tensor:
        device = next(player.backend.model.parameters()).device
        policy_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)

        for i, response in enumerate(responses):
            log_prob_old = response["seq_logprob"].detach().to(device)

            # TODO: remove no_grad when optimizer is wired up
            with torch.no_grad():
                log_prob_theta = self._compute_seq_logprob(
                    player.backend.model,
                    prompt_input_ids=response["prompt_input_ids"],
                    prompt_attention_mask=response["prompt_attention_mask"],
                    new_tokens=response["new_tokens"],
                )

            with torch.no_grad():
                log_prob_ref = self._compute_seq_logprob(
                    self.reference_model,
                    prompt_input_ids=response["prompt_input_ids"],
                    prompt_attention_mask=response["prompt_attention_mask"],
                    new_tokens=response["new_tokens"],
                ).to(device)

            ratio = torch.exp(log_prob_theta - log_prob_old)
            adv = advantages[i].to(device)

            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            policy_loss = policy_loss - torch.min(ratio * adv, clipped * adv)

            kl = torch.exp(log_prob_ref - log_prob_theta) - log_prob_ref + log_prob_theta - 1
            kl_loss = kl_loss + kl

        return (policy_loss + beta * kl_loss) / len(responses)

    def _compute_seq_logprob(
        self,
        model: torch.nn.Module,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        new_tokens: torch.Tensor,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        prompt_len = prompt_input_ids.shape[1]

        full_ids = torch.cat(
            [prompt_input_ids.to(device), new_tokens.to(device)], dim=1
        )
        full_mask = torch.cat(
            [prompt_attention_mask.to(device),
            torch.ones(1, new_tokens.shape[1], device=device, dtype=torch.long)],
            dim=1,
        )

        outputs = model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits

        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = new_tokens.to(device)

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum(dim=-1)
    
    def run(self, num_steps: int = 1):
        """Run the game for num_turns."""
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                break

    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig]):
        """Create an arena from a config."""
        if isinstance(config, str):
            config = ArenaConfig.load(config)

        global_prompt = config.get("global_prompt", None)

        # Inject global prompt into each player config
        player_configs = []
        for player_config in config.players:
            player_config = dict(player_config)
            if global_prompt is not None:
                player_config["global_prompt"] = global_prompt
            player_configs.append(player_config)

        # Let the environment create and manage players
        config.environment["player_configs"] = player_configs

        env = load_environment(config.environment)

        clue_number = config.get("clue_number", 3)
        num_grpo_epochs = config.get("num_grpo_epochs", 10)
        return cls(env, global_prompt=global_prompt, clue_number=clue_number, num_grpo_epochs=num_grpo_epochs)

    def to_config(self) -> ArenaConfig:
        """Convert the arena to a config."""
        return ArenaConfig(
            environment=self.environment.to_config(),
            global_prompt=self.global_prompt,
        )

    def launch_cli(self, max_steps: int = None, interactive: bool = True):
        """Launch the command line interface."""
        from .ui.cli import ArenaCLI

        cli = ArenaCLI(self)
        cli.launch(max_steps=max_steps, interactive=interactive)

    def save_config(self, path: str):
        """Save the config to a file."""
        config = self.to_config()
        config.save(path)

    def save_history(self, path: str):
        """
        Save the history of the game to a file.

        Supports csv and json formats.
        """
        messages = self.environment.get_observation()
        message_rows = []

        if path.endswith(".csv"):
            header = [
                "agent_name",
                "content",
                "turn",
                "timestamp",
                "visible_to",
                "msg_type",
            ]
            for message in messages:
                message_row = [
                    message.agent_name,
                    message.content,
                    message.turn,
                    str(message.timestamp),
                    message.visible_to,
                    message.msg_type,
                ]
                message_rows.append(message_row)

            with open(path, "w") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(message_rows)
        elif path.endswith(".json"):
            for message in messages:
                message_row = {
                    "agent_name": message.agent_name,
                    "content": message.content,
                    "turn": message.turn,
                    "timestamp": str(message.timestamp),
                    "visible_to": message.visible_to,
                    "msg_type": message.msg_type,
                }
                message_rows.append(message_row)

            with open(path, "w") as f:
                json.dump(message_rows, f, indent=4)
        else:
            raise ValueError("Invalid file format")

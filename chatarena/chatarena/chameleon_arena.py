import atexit
import csv
import json
import logging
import os
import uuid
from datetime import datetime
import torch
from typing import Dict, List, Union

from .chameleon_agent import Player
from .config import ArenaConfig
from .environments import Environment, TimeStep, load_environment, Chameleon

WIDTH = 80


class TooManyInvalidActions(Exception):
    pass


class RunLogger:
    def __init__(self, log_dir: str = "logs", log_path: str | None = None):
        os.makedirs(log_dir, exist_ok=True)
        if log_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(log_dir, f"run_{timestamp}.log")
        else:
            self.path = log_path
        self._file = open(self.path, "a")
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
        best_action: str,
        new_messages: list,
        terminal_rewards: dict,
        responses: list = None,
        belief_rewards: list = None,
        advantages: list = None,
        best_idx: int = 0,
        grpo_losses: list = None,
        belief_loss: float = None,
        post_clue_beliefs: dict = None,
    ):
        self._write()
        self._write(f"  [ {player_name} ]")

        if belief_rewards is not None and advantages is not None:
            for i, (resp, reward, adv) in enumerate(zip(responses, belief_rewards, advantages)):
                action_preview = resp["action"].replace("\n", " ").strip()[:120]
                self._write(f"    Attempt {i + 1}  belief_reward={reward:+.4f}  advantage={adv:+.4f}")
                self._write(f"      \"{action_preview}\"")
            self._write(f"    >> Best attempt {best_idx + 1}: \"{best_action.replace(chr(10), ' ').strip()[:120]}\"")

            rewards_t = torch.as_tensor(belief_rewards, dtype=torch.float32)
            mean = rewards_t.mean().item()
            std = rewards_t.std().item() if rewards_t.numel() > 1 else 0.0
            chosen = belief_rewards[best_idx]
            self._write(
                f"    Belief reward: mean={mean:+.4f}  std={std:.4f}  chosen={chosen:+.4f}"
            )
        else:
            action_preview = best_action.replace("\n", " ").strip()[:120]
            self._write(f"    Action: \"{action_preview}\"")

        if grpo_losses:
            loss_parts = "  ".join(f"e{i+1}:{l:.4f}" for i, l in enumerate(grpo_losses))
            self._write(f"    GRPO losses: {loss_parts}")

        if belief_loss is not None:
            self._write(f"    Belief CE loss: {belief_loss:.4f}")

        if post_clue_beliefs:
            self._write(f"    Post-clue beliefs:")
            for pname, dist in post_clue_beliefs.items():
                sorted_d = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                parts = [f"{n}:{p:.3f}" for n, p in sorted_d if p > 1e-3]
                self._write(f"      {pname}: [{', '.join(parts)}]")

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

    def log_vote(
        self,
        player_name: str,
        belief_distribution: dict,
        voted_for: str,
        new_messages: list,
        terminal_rewards: dict,
    ):
        self._write()
        self._write(f"  [ {player_name} — VOTE (belief-based) ]")

        sorted_beliefs = sorted(belief_distribution.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_beliefs:
            marker = " <<" if name == voted_for else ""
            self._write(f"    {name}: {prob:.4f}{marker}")

        self._write(f"    >> Voted for: {voted_for}")

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



def _get_action(response):
    """Extract action string from either dict (HuggingFace) or str (API) response."""
    if isinstance(response, dict):
        return response["action"]
    return response


class ChameleonArena:
    """Utility class that manages the game environment and players."""

    def __init__(
        self, environment: Chameleon, global_prompt: str = None, clue_number: int = 3, num_grpo_epochs: int = 3, policy_lr: int = 1e-4, belief_lr: int = 1e-5,
        logger: RunLogger | None = None,
    ):
        # Create a container for the players and environment and reset the game
        self.environment = environment
        self.global_prompt = global_prompt
        self.clue_number = clue_number

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.num_grpo_epochs = num_grpo_epochs

        # GRPO policy training is only possible with a trainable HuggingFace backend
        self.train_policy = (
            self.environment.backend is not None
            and hasattr(self.environment.backend, 'model')
        )

        if self.train_policy:
            self.policy_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.environment.backend.model.parameters()),
                lr=policy_lr,
            )
        else:
            self.reference_model = None
            self.policy_optimizer = None

        self.logger = logger if logger is not None else RunLogger()
        self.logger.log_game_start(
            topic=self.environment.topic,
            code=self.environment.code,
            chameleon_name=self.environment.chameleon_name,
            player_names=self.environment.player_names,
        )

        self.belief_optimizer = None


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

    def _collect_non_chameleon_beliefs(self) -> dict:
        env = self.environment
        if getattr(env, "player_belief", None) is None:
            return {}
        beliefs = env.player_belief.detach()
        shared = {
            env.player_names[i]: beliefs[i].item()
            for i in range(len(env.player_names))
        }
        return {"shared": shared}

    def step(self) -> TimeStep:
        """Take a step in the game: one player takes an action and the environment updates."""
        env = self.environment
        player_name = env.get_next_player()
        player = self.name_to_player[player_name]
        observation = env.get_observation(player_name)

        if env._current_phase == "give clues" and player_name != env.chameleon_name and self.train_policy:
            responses = [player(observation) for _ in range(self.clue_number)]
            clues = [r["action"] for r in responses]

            with torch.no_grad():
                rewards = env.evaluate_clues(player_name, clues)

            rewards = [
                r.item() if isinstance(r, torch.Tensor) else float(r)
                for r in rewards
            ]

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
            advantages = (rewards_tensor - rewards_tensor.mean()) / (
                rewards_tensor.std(unbiased=False) + 1e-8
            )

            grpo_losses = []
            for _ in range(self.num_grpo_epochs):
                loss = self._compute_grpo_loss(player, responses, advantages)
                grpo_losses.append(loss.item())
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()

            best_idx = int(advantages.argmax().item())
            best_action = responses[best_idx]["action"]

            msg_count_before = len(env.message_pool._messages)
            timestep = env.step(player_name, best_action)
            new_messages = env.message_pool._messages[msg_count_before:]

            self.logger.log_step(
                player_name=player_name,
                responses=responses,
                belief_rewards=rewards,
                advantages=advantages.tolist(),
                best_idx=best_idx,
                best_action=best_action,
                grpo_losses=grpo_losses,
                post_clue_beliefs=self._collect_non_chameleon_beliefs(),
                new_messages=new_messages,
                terminal_rewards=timestep.reward if timestep.terminal else None,
            )

        elif env._current_phase == "give clues":
            response = player(observation)
            action = _get_action(response)

            with torch.no_grad():
                env.evaluate_clues(player_name, [action])

            msg_count_before = len(env.message_pool._messages)
            timestep = env.step(player_name, action)
            new_messages = env.message_pool._messages[msg_count_before:]

            self.logger.log_step(
                player_name=player_name,
                best_action=action,
                post_clue_beliefs=self._collect_non_chameleon_beliefs(),
                new_messages=new_messages,
                terminal_rewards=timestep.reward if timestep.terminal else None,
            )

        elif env._current_phase == "accuse":
            voted_player = player.vote_from_belief(env.player_belief)
            action = f"I vote for {voted_player}."

            msg_count_before = len(env.message_pool._messages)
            timestep = env.step(player_name, action)
            new_messages = env.message_pool._messages[msg_count_before:]

            if player_name != env.chameleon_name:
                beliefs = env.player_belief.detach()
                belief_dict = {
                    env.player_names[i]: beliefs[i].item()
                    for i in range(len(env.player_names))
                }
                self.logger.log_vote(
                    player_name=player_name,
                    belief_distribution=belief_dict,
                    voted_for=voted_player,
                    new_messages=new_messages,
                    terminal_rewards=timestep.reward if timestep.terminal else None,
                )
            else:
                self.logger.log_step(
                    player_name=player_name,
                    best_action=action,
                    new_messages=new_messages,
                    terminal_rewards=timestep.reward if timestep.terminal else None,
                )

        elif env._current_phase == "guess":
            guessed_word = player.guess_from_belief(env.word_belief)
            action = f"I guess the secret word is {guessed_word}."

            msg_count_before = len(env.message_pool._messages)
            timestep = env.step(player_name, action)
            new_messages = env.message_pool._messages[msg_count_before:]

            self.logger.log_step(
                player_name=player_name,
                best_action=action,
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
        beta: float = 0.2,
    ) -> torch.Tensor:
        device = next(player.backend.model.parameters()).device
        policy_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)

        for i, response in enumerate(responses):
            seq_len = response["new_tokens"].shape[1]
            log_prob_old = response["seq_logprob"].detach().to(device)

            log_prob_theta = self._compute_seq_logprob(
                player.backend.model,
                prompt_input_ids=response["prompt_input_ids"],
                prompt_attention_mask=response["prompt_attention_mask"],
                new_tokens=response["new_tokens"],
            )

            ratio = torch.exp((log_prob_theta - log_prob_old))
            adv = advantages[i].to(device)

            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            policy_loss = policy_loss - torch.min(ratio * adv, clipped * adv) / seq_len

            with player.backend.model.disable_adapter():
                log_prob_ref = self._compute_seq_logprob(
                    player.backend.model,
                    prompt_input_ids=response["prompt_input_ids"],
                    prompt_attention_mask=response["prompt_attention_mask"],
                    new_tokens=response["new_tokens"],
                )
                
            kl = torch.exp(log_prob_ref - log_prob_theta) - log_prob_ref + log_prob_theta - 1
            kl_loss = kl_loss + kl / seq_len

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
        """Run the game for num_turns. Returns the final timestep."""
        timestep = None
        for i in range(num_steps):
            timestep = self.step()
            if timestep.terminal:
                break
        return timestep

    @classmethod
    def from_config(cls, config: Union[str, ArenaConfig], logger: RunLogger | None = None):
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
        return cls(env, global_prompt=global_prompt, clue_number=clue_number, num_grpo_epochs=num_grpo_epochs, logger=logger)

    def to_config(self) -> ArenaConfig:
        """Convert the arena to a config."""
        return ArenaConfig(
            environment=self.environment.to_config(),
            global_prompt=self.global_prompt,
        )

    def launch_cli(self, max_steps: int = None, interactive: bool = True):
        """Launch the command line interface."""
        from .ui.chamelon_cli import ChameleonArenaCLI

        cli = ChameleonArenaCLI(self)
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

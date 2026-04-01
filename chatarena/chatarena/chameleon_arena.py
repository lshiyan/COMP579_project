import csv
import json
import logging
import uuid
from typing import Dict, List, Union

from .agent import Player
from .config import ArenaConfig
from .environments import Environment, TimeStep, load_environment, Chameleon


class TooManyInvalidActions(Exception):
    pass


class ChameleonArena:
    """Utility class that manages the game environment and players."""

    def __init__(
        self, environment: Chameleon, global_prompt: str = None
    ):
        # Create a container for the players and environment and reset the game      
        self.environment = environment
        self.global_prompt = global_prompt

        self.current_timestep = environment.reset()
        self.uuid = uuid.uuid4()  # Generate a unique id for the game
        self.invalid_actions_retry = 5

    @property
    def num_players(self):
        return self.environment.num_players

    @property
    def name_to_player(self) -> Dict[str, Player]:
        return {player.name: player for player in self.environment.players}

    def reset(self) -> TimeStep:
        # Reset the environment
        self.current_timestep = self.environment.reset()
        # Reset the uuid
        self.uuid = uuid.uuid4()
        return self.current_timestep

    def step(self) -> TimeStep:
        """Take a step in the game: one player takes an action and the environment updates."""
        player_name = self.environment.get_next_player()
        player = self.name_to_player[player_name]  # get the player object
        observation = self.environment.get_observation(
            player_name
        )  # get the observation for the player

        timestep = None
        for i in range(
            self.invalid_actions_retry
        ):  # try to take an action for a few times
            
            action, latent_state = player(observation)  # take an action
            
            print(action)
            if self.environment.check_action(action, player_name):  # action is valid
                timestep = self.environment.step(
                    player_name, action
                )  # update the environment
                break
            else:  # action is invalid
                logging.warning(f"{player_name} made an invalid action {action}")
                continue

        if (
            timestep is None
        ):  # if the player made invalid actions for too many times, terminate the game
            warning_msg = f"{player_name} has made invalid actions for {self.invalid_actions_retry} times. Terminating the game."
            logging.warning(warning_msg)
            raise TooManyInvalidActions(warning_msg)

        return timestep

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

        return cls(env, global_prompt=global_prompt)

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

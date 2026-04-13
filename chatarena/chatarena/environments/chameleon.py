import json
import random
import re
from typing import Dict, List, Optional, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

DEFAULT_TOPIC_CODES = {
    "Fruits": [
        "Apple", "Banana", "Orange", "Grape",
        "Strawberry", "Pineapple", "Mango", "Watermelon",
    ],
    "Animals": [
        "Lion", "Elephant", "Giraffe", "Monkey",
        "Zebra", "Tiger", "Bear", "Kangaroo",
    ],
    "Sports": [
        "Soccer", "Basketball", "Tennis", "Baseball",
        "Swimming", "Cycling", "Volleyball", "Golf",
    ],
    "Countries": [
        "United States", "Canada", "Brazil", "United Kingdom",
        "France", "Germany", "Japan", "Australia",
    ],
}


@register_env
class Chameleon(Environment):
    type_name = "chameleon"

    def __init__(self, player_names: List[str], topic_codes: Dict[str, List[str]] = None, **kwargs):
        super().__init__(player_names=player_names, topic_codes=topic_codes, **kwargs)

        if topic_codes is None:
            topic_codes = DEFAULT_TOPIC_CODES
        self.topic_codes = topic_codes

        self.message_pool = MessagePool()

        self.topic = None
        self.code = None
        self.chameleon_name = None
        self.non_chameleon_names = None

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"
        self._players_votes = None
        self._initialized = False

        self.reset()

    def get_next_player(self) -> str:
        if self._current_phase != "guess":
            return self.player_names[self._next_player_idx]
        return self.chameleon_name

    def reset(self):
        self.topic = random.choice(list(self.topic_codes.keys()))
        self.code = random.choice(self.topic_codes[self.topic])
        self.chameleon_name = random.choice(self.player_names)
        self.non_chameleon_names = [
            name for name in self.player_names if name != self.chameleon_name
        ]

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"

        self.message_pool.reset()

        self._moderator_speak(f"Now the game starts! The topic is: {self.topic}")
        self._moderator_speak(
            f"You are not chameleon. The word is: {self.code}",
            visible_to=self.non_chameleon_names,
        )
        self._moderator_speak("You are the chameleon!", visible_to=self.chameleon_name)
        self._moderator_speak(
            "Now everyone gives one clue (but don't give away the secret word). "
            f"You cannot repeat what others has said. We will start with {self.player_names[0]}."
        )
        self._current_turn = 1
        self._players_votes = {name: 0 for name in self.player_names}
        self._initialized = True

        return TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        return self.message_pool.get_visible_messages(player_name, turn=self._current_turn)

    def _text2vote(self, text) -> str:
        text = text.lower()
        for name in self.player_names:
            candidates = [
                name.lower(),
                name.lower().replace(" ", ""),
                name.lower().replace(" ", "_"),
            ]
            if any(candidate in text for candidate in candidates):
                return name
        return ""
    
    def _is_true_code(self, text) -> bool:
        """Check whether the text is the true code."""
        # Get the word enclosed by quote marks with regex
        pattern = r"\"(.+?)\""
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower().replace(" ", "") == self.code.lower().replace(
                " ", ""
            )
        else:
            # if no quote marks, check whether the last k words match the code
            words = text.split()
            if len(words) >= len(self.code.split()):
                guessed_term = (
                    "".join(words[-len(self.code.split()) :]).lower().replace(".", "")
                )
                return guessed_term == self.code.lower().replace(" ", "").replace(
                    ".", ""
                )
            else:
                return False

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def get_rewards(self, chameleon_win: bool) -> Dict[str, float]:
        rewards = {}
        for name in self.player_names:
            rewards[name] = float((name == self.chameleon_name) == chameleon_win)
        return rewards

    def is_terminal(self) -> bool:
        if self.message_pool.last_message.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        return False

    def step(self, player_name: str, action: str) -> TimeStep:
        if not self._initialized:
            self.reset()

        assert player_name == self.get_next_player(), (
            f"Wrong player! It is {self.get_next_player()} turn."
        )

        if self._current_phase == "give clues":
            message = Message(agent_name=player_name, content=action, turn=self._current_turn)
            self.message_pool.append_message(message)

            self._current_turn += 1
            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_phase = "accuse"
                self._moderator_speak(
                    "Now vote which of the other players (excluding yourself) is the chameleon. "
                    "You cannot vote for yourself."
                )
                self._current_turn += 1

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=self.get_zero_rewards(),
                terminal=False,
            )

        elif self._current_phase == "accuse":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name],
            )
            self.message_pool.append_message(message)

            vote = self._text2vote(action)
            if vote in self.player_names:
                self._players_votes[vote] += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_zero_rewards(),
                    terminal=False,
                )
            else:
                accuse_correct, even_vote = True, False
                max_vote_player = max(self._players_votes, key=self._players_votes.get)
                # detach if other players has the same number of votes
                for name, vote in self._players_votes.items():
                    if (
                        name != max_vote_player
                        and vote == self._players_votes[max_vote_player]
                    ):
                        accuse_correct, even_vote = False, True
                if max_vote_player != self.chameleon_name:
                    accuse_correct = False

                if not accuse_correct:
                    if even_vote:
                        self._moderator_speak(
                            f"There are even votes. The accusation does not stand. "
                            f"{self.chameleon_name} is the chameleon. {self.chameleon_name} won the game!"
                        )
                    else:
                        self._moderator_speak(
                            f"The most-voted player is {max_vote_player}. The accusation is incorrect. "
                            f"{self.chameleon_name} is the chameleon. {self.chameleon_name} won the game!"
                        )
                    rewards = self.get_rewards(chameleon_win=True)
                    terminal = True
                else:
                    self._moderator_speak(
                        f"The accusation is correct! {self.chameleon_name} is the chameleon! "
                        f"Now {self.chameleon_name} can guess the secret code. "
                        'You should say: I guess the code is "..."'
                    )
                    self._current_phase = "guess"
                    rewards = self.get_zero_rewards()
                    terminal = False
                self._current_turn += 1

                timestep = TimeStep(
                    observation=self.get_observation(), reward=rewards, terminal=terminal
                )
        elif self._current_phase == "guess":
            print(action)
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=player_name,
            )
            self.message_pool.append_message(message)
            if self._is_true_code(action):
                self._moderator_speak(
                    f"{player_name} guessed the code correctly! The secret word is {self.code}. "
                    f"{self.chameleon_name} won!"
                )
                
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_rewards(chameleon_win=True),
                    terminal=True,
                    chameleon_won=True,
                )
            else:
                self._moderator_speak(
                    f"{player_name} guessed the code wrong! The secret word is {self.code}. "
                    f"{self.non_chameleon_names} won!"
                )
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_rewards(chameleon_win=False),
                    terminal=True,
                    chameleon_won=False,
                )

        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        if self.is_terminal():
            timestep.terminal = True

        return timestep
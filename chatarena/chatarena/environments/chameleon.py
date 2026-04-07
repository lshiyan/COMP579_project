import random
import re
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..agent import SIGNAL_END_OF_CONVERSATION, Player
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from ..backends import TransformersLlamaChat

DEFAULT_TOPIC_CODES = {
    "Fruits": ["Apple", "Banana", "Orange", "Grape", "Strawberry", "Pineapple", "Mango", "Watermelon"],
    "Animals": ["Lion", "Elephant", "Giraffe", "Monkey", "Zebra", "Tiger", "Bear", "Kangaroo"],
    "Sports": ["Soccer", "Basketball", "Tennis", "Baseball", "Swimming", "Cycling", "Volleyball", "Golf"],
    "Countries": ["United States", "Canada", "Brazil", "United Kingdom", "France", "Germany", "Japan", "Australia"],
    "Food": ["pizza", "sushi", "hamburger", "pasta", "taco", "croissant", "ramen", "curry"],
    "Movies": ["Titanic", "Inception", "Avatar", "Jaws", "Psycho", "Casablanca", "Alien", "Grease"],
}

@register_env
class Chameleon(Environment):
    type_name = "chameleon"
    backend: TransformersLlamaChat

    def __init__(
        self,
        player_configs: List[dict],
        backend: TransformersLlamaChat,
        embedding_size: int = 384, #For clue embeddings
        belief_state_size: int = 512, #Belief state size
        speaker_embedding_size: int = 64,
        role_embedding_size: int = 16,
        num_clue_rounds: int = 1,
        **kwargs,
    ):
        self.topic_codes = DEFAULT_TOPIC_CODES

        self.embedding_size = embedding_size
        self.belief_state_size = belief_state_size
        self.speaker_embedding_size = speaker_embedding_size
        self.role_embedding_size = role_embedding_size

        if num_clue_rounds < 1:
            raise ValueError("num_clue_rounds must be >= 1")
        self.num_clue_rounds = num_clue_rounds

        if isinstance(backend, TransformersLlamaChat):
            self.backend: TransformersLlamaChat = backend
        else:
            self.backend = TransformersLlamaChat.from_config(backend)

        max_num_players = len(player_configs)
        max_num_words = max(len(words) for words in self.topic_codes.values())

        llm_device = next(self.backend.model.parameters()).device

        self.shared_belief_updater = nn.GRUCell(
            input_size=(
                self.embedding_size
                + self.speaker_embedding_size
                + self.role_embedding_size
            ),
            hidden_size=self.belief_state_size,
        ).to(llm_device)
        self.shared_speaker_embedding = nn.Embedding(
            max_num_players, self.speaker_embedding_size
        ).to(llm_device)
        self.shared_role_embedding = nn.Embedding(2, self.role_embedding_size).to(llm_device)

        self.shared_player_belief_head = nn.Linear(
            self.belief_state_size, max_num_players
        ).to(llm_device)
        self.shared_word_belief_head = nn.Linear(
            self.belief_state_size, max_num_words
        ).to(llm_device)

        self.players = [
            Player(
                name=cfg["name"],
                role_desc=cfg["role_desc"],
                backend=self.backend,
                global_prompt=cfg.get("global_prompt", None),
                embedding_size=self.embedding_size,
                belief_state_size=self.belief_state_size,
                shared_player_belief_head=self.shared_player_belief_head,
                shared_word_belief_head=self.shared_word_belief_head,
                shared_belief_updater=self.shared_belief_updater,
                shared_speaker_embedding=self.shared_speaker_embedding,
                shared_role_embedding=self.shared_role_embedding,
            )
            for cfg in player_configs
        ]

        self.player_names = [player.name for player in self.players]
        self.agent_to_idx = {name: i for i, name in enumerate(self.player_names)}

        super().__init__(
            player_names=self.player_names,
            topic_codes=self.topic_codes,
            **kwargs,
        )

        self.message_pool = MessagePool()

        self.topic = None
        self.code = None
        self.chameleon_name = None
        self.non_chameleon_names = None
        self.word_to_idx = None

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"
        self._current_clue_round = 0
        self._players_votes = None
        self._initialized = False

        self.rewards = []

        self.reset()

    def get_next_player(self) -> str:
        if self._current_phase != "guess":
            return self.player_names[self._next_player_idx]
        assert self.chameleon_name is not None
        return self.chameleon_name

    def reset(self):
        self.topic = random.choice(list(self.topic_codes.keys()))
        self.code = random.choice(self.topic_codes[self.topic])
        self.chameleon_name = random.choice(self.player_names)
        self.non_chameleon_names = [
            name for name in self.player_names if name != self.chameleon_name
        ]

        current_words = self.topic_codes[self.topic]
        self.word_to_idx = {word: i for i, word in enumerate(current_words)}

        for player in self.players:
            player.set_shared_belief_heads(
                self.shared_player_belief_head,
                self.shared_word_belief_head,
            )
            player.set_shared_belief_modules(
                self.shared_belief_updater,
                self.shared_speaker_embedding,
                self.shared_role_embedding,
            )

            if player.name != self.chameleon_name:
                player.set_hidden_role(
                    "non_chameleon",
                    self.player_names,
                    current_words,
                )
            else:
                player.set_hidden_role(
                    "chameleon",
                    self.player_names,
                    current_words,
                )

        self._current_turn = 0
        self._next_player_idx = 0
        self._current_phase = "give clues"
        self._current_clue_round = 0

        self.message_pool.reset()

        self._moderator_speak(f"Now the game starts! The topic is: {self.topic}")
        self._moderator_speak(
            f"You are not chameleon. The word is: {self.code}",
            visible_to=self.non_chameleon_names,
        )
        self._moderator_speak(
            "You are the chameleon!",
            visible_to=self.chameleon_name,
        )
        self._moderator_speak(
            f"Now everyone gives {self.num_clue_rounds} clue round(s) "
            f"(but don't give away the secret word). "
            f"You cannot repeat what others have said. "
            f"We will start with {self.player_names[0]}. "
            f"Round 1/{self.num_clue_rounds}."
        )
        self._current_turn = 1

        self._players_votes = {name: 0 for name in self.player_names}
        self._initialized = True
        self.rewards = []

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
        return self.message_pool.get_visible_messages(
            player_name,
            turn=self._current_turn,
        )

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
        assert self.code is not None
        pattern = r"\"(.+?)\""
        match = re.search(pattern, text)
        if match:
            return (
                match.group(1).lower().replace(" ", "")
                == self.code.lower().replace(" ", "")
            )
        words = text.split()
        if len(words) >= len(self.code.split()):
            guessed_term = "".join(words[-len(self.code.split()):]).lower().replace(".", "")
            return guessed_term == self.code.lower().replace(" ", "").replace(".", "")
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
        last = self.message_pool.last_message
        if last is not None and last.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        return False

    def _encode_message_for_beliefs(self, action: str) -> torch.Tensor:
        message_embedding = self.backend.get_message_embedding(
            message_text=action,
        )

        if message_embedding.dim() == 1:
            message_embedding = message_embedding.unsqueeze(0)

        return message_embedding

    def _update_beliefs_for_new_clue(self, speaker_name: str, action: str):
        message_embedding = self._encode_message_for_beliefs(action)

        for player in self.players:
            player.update_belief_state(
                message_embedding=message_embedding,
                speaker_name=speaker_name,
            )

    def _compute_belief_reward(
        self,
        speaker_name,
        prior_beliefs,
        post_beliefs,
        alpha=0.5,
    ):
        speaker_idx = self.agent_to_idx[speaker_name]
        chameleon_name = self.chameleon_name
        word_idx = self.word_to_idx[self.code]

        suspicion_terms = []
        for player in self.players:
            if player.name != speaker_name and player.name != chameleon_name:
                prior_prob = prior_beliefs[player.name][speaker_idx]
                post_prob = post_beliefs[player.name][speaker_idx]
                suspicion_terms.append(prior_prob - post_prob)

        if len(suspicion_terms) > 0:
            suspicion_reward = torch.stack(suspicion_terms).mean()
        else:
            suspicion_reward = torch.tensor(0.0)

        prior_word_prob = prior_beliefs[chameleon_name][word_idx]
        post_word_prob = post_beliefs[chameleon_name][word_idx]

        if speaker_name == chameleon_name:
            word_reward = post_word_prob - prior_word_prob
        else:
            word_reward = prior_word_prob - post_word_prob

        belief_reward = alpha * suspicion_reward + (1 - alpha) * word_reward
        return belief_reward

    def evaluate_clue(self, speaker_name: str, action: str, update = False):
        prior_beliefs = {}
        prior_belief_states = {}

        for player in self.players:
            belief, belief_state = player.save_beliefs()
            prior_beliefs[player.name] = belief.detach()
            prior_belief_states[player.name] = belief_state.detach()

        self._update_beliefs_for_new_clue(
            speaker_name=speaker_name,
            action=action,
        )

        post_beliefs = {
            player.name: player.beliefs.clone()
            for player in self.players
        }

        if speaker_name != self.chameleon_name:
            belief_reward = self._compute_belief_reward(
                speaker_name, prior_beliefs, post_beliefs
            )
        else:
            belief_reward = torch.tensor(0.0)

        if not update:
            for player in self.players:
                player.set_beliefs(
                    prior_beliefs[player.name],
                    prior_belief_states[player.name],
                )

        return belief_reward
                    
    def step(self, player_name: str, action: str) -> TimeStep:
        if not self._initialized:
            self.reset()

        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()} turn."

        if self._current_phase == "give clues":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
            )
            
            self.message_pool.append_message(message)
            
            self._current_turn += 1

            if self._next_player_idx < len(self.player_names) - 1:
                self._next_player_idx += 1
            else:
                self._next_player_idx = 0
                self._current_clue_round += 1

                if self._current_clue_round < self.num_clue_rounds:
                    self._moderator_speak(
                        f"Clue round {self._current_clue_round + 1}/{self.num_clue_rounds}. "
                        f"We will start with {self.player_names[0]} again."
                    )
                    self._current_turn += 1
                else:
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
                rewards = self.get_zero_rewards()
                terminal = False
            else:
                accuse_correct, even_vote = True, False
                max_vote_player = max(self._players_votes, key=self._players_votes.get)

                for name, vote_count in self._players_votes.items():
                    if (
                        name != max_vote_player
                        and vote_count == self._players_votes[max_vote_player]
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
                observation=self.get_observation(),
                reward=rewards,
                terminal=terminal,
            )

        elif self._current_phase == "guess":
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
                rewards = self.get_rewards(chameleon_win=True)
            else:
                self._moderator_speak(
                    f"{player_name} guessed the code wrong! The secret word is {self.code}. "
                    f"{self.non_chameleon_names} won!"
                )
                rewards = self.get_rewards(chameleon_win=False)

            timestep = TimeStep(
                observation=self.get_observation(),
                reward=rewards,
                terminal=True,
            )

        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        if self.is_terminal():
            timestep.terminal = True

        return timestep
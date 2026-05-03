import random
import re
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import math

from ..backends import TransformersHuggingFaceChat
from ..chameleon_agent import SIGNAL_END_OF_CONVERSATION, Player
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

DEFAULT_TOPIC_CODES = {
    "Fruits": [
        "Apple",
        "Banana",
        "Orange",
        "Grape",
        "Strawberry",
        "Pineapple",
        "Mango",
        "Watermelon",
    ],
    "Animals": [
        "Lion",
        "Elephant",
        "Giraffe",
        "Monkey",
        "Zebra",
        "Tiger",
        "Bear",
        "Kangaroo",
    ],
    "Sports": [
        "Soccer",
        "Basketball",
        "Tennis",
        "Baseball",
        "Swimming",
        "Cycling",
        "Volleyball",
        "Golf",
    ],
    "Food": [
        "pizza",
        "sushi",
        "hamburger",
        "pasta",
        "taco",
        "croissant",
        "ramen",
        "curry",
    ],
}


@register_env
class Chameleon(Environment):
    type_name = "chameleon_grpo"
    backend: TransformersHuggingFaceChat

    def __init__(
        self,
        player_configs: List[dict],
        backend=None,
        sentence_encoder=None,
        embedding_size: int = 384,  # For clue embeddings
        belief_state_size: int = 512,  # Belief state size
        speaker_embedding_size: int = 64,
        num_clue_rounds: int = 1,
        reward_alpha: float = 0.5,
        reward_gamma: float = 2.0,
        reward_word_leak_threshold: float = 0.15,
        reward_max_tokens: int = 12,
        reward_zeta: float = 0.1,
        reward_length_cap: float = 2.0,
        reward_lmb: float = 1.0,
        reward_eta: float = 1.0,
        **kwargs,
    ):
        self.topic_codes = DEFAULT_TOPIC_CODES

        self.embedding_size = embedding_size
        self.belief_state_size = belief_state_size
        self.speaker_embedding_size = speaker_embedding_size

        if num_clue_rounds < 1:
            raise ValueError("num_clue_rounds must be >= 1")
        self.num_clue_rounds = num_clue_rounds

        self.reward_alpha = reward_alpha
        self.reward_gamma = reward_gamma
        self.reward_word_leak_threshold = reward_word_leak_threshold
        self.reward_max_tokens = reward_max_tokens
        self.reward_zeta = reward_zeta
        self.reward_length_cap = reward_length_cap
        self.reward_lmb = reward_lmb
        self.reward_eta = reward_eta

        # Backend for LLM generation (can be HuggingFace or None for CS)
        if isinstance(backend, TransformersHuggingFaceChat):
            self.backend = backend
        elif backend is not None:
            self.backend = TransformersHuggingFaceChat.from_config(backend)
        else:
            self.backend = None

        # Standalone sentence encoder (used when backend is not HuggingFace)
        self._sentence_encoder = sentence_encoder

        max_num_players = len(player_configs)
        max_num_words = max(len(words) for words in self.topic_codes.values())

        # Determine device for belief network modules
        if self.backend is not None:
            self.belief_device = next(self.backend.model.parameters()).device
        elif self._sentence_encoder is not None and hasattr(
            self._sentence_encoder, "device"
        ):
            self.belief_device = self._sentence_encoder.device
        else:
            self.belief_device = torch.device("cpu")

        self.players = [
            Player(
                name=cfg["name"],
                role_desc=cfg["role_desc"],
                backend=self.backend if self.backend is not None else cfg["backend"],
                global_prompt=cfg.get("global_prompt", None),
                embedding_size=self.embedding_size,
                belief_state_size=self.belief_state_size,
            )
            for cfg in player_configs
        ]

        self.player_names = [player.name for player in self.players]
        self.agent_to_idx = {name: i for i, name in enumerate(self.player_names)}
        self.topic_to_idx = {
            topic: i for i, topic in enumerate(self.topic_codes.keys())
        }

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

        num_players = len(self.player_names)
        num_words = len(current_words)

        self.player_belief = (
            torch.ones(
                num_players,
                dtype=torch.float32,
                device=self.belief_device,
            )
            / num_players
        )

        self.word_belief = (
            torch.ones(
                num_words,
                dtype=torch.float32,
                device=self.belief_device,
            )
            / num_words
        )

        for player in self.players:
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

        clue_message = (
            f"Now everyone gives {self.num_clue_rounds} clue round(s) "
            f"(but don't give away the secret word). "
            f"You cannot repeat what others have said.\n\n"
            f"IMPORTANT RULE:\n"
            f"Your clue MUST contain AT MOST 10 words.\n"
            f"If your response exceeds 10 words, it will be considered INVALID.\n"
            f"Output ONLY the clue. Do not explain.\n\n"
            f"We will start with {self.player_names[0]}. "
            f"Round 1/{self.num_clue_rounds}."
        )

        self._moderator_speak(clue_message)
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

    def get_reward_weights(self) -> dict:
        return {
            "alpha": self.reward_alpha,
            "gamma": self.reward_gamma,
            "word_leak_threshold": self.reward_word_leak_threshold,
            "max_tokens": self.reward_max_tokens,
            "zeta": self.reward_zeta,
            "length_cap": self.reward_length_cap,
            "lmb": self.reward_lmb,
            "eta": self.reward_eta,
        }

    def get_observation(self, player_name=None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        return self.message_pool.get_visible_messages(
            player_name,
            turn=self._current_turn,
        )

    def get_votes(self):
        return self._players_votes

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
            return match.group(1).lower().replace(" ", "") == self.code.lower().replace(
                " ", ""
            )
        words = text.split()
        if len(words) >= len(self.code.split()):
            guessed_term = (
                "".join(words[-len(self.code.split()) :]).lower().replace(".", "")
            )
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

    def evaluate_clues(
        self,
        speaker_name: str,
        clues: List[str],
        lmb: float = None,
        eta: float = None,
        alpha: float = None,
        gamma: float = None,
        max_tokens: int = None,
        zeta: float = None,
        word_leak_threshold: float = None,
        length_cap: float = None,
    ):
        if lmb is None: lmb = self.reward_lmb
        if eta is None: eta = self.reward_eta
        if alpha is None: alpha = self.reward_alpha
        if gamma is None: gamma = self.reward_gamma
        if max_tokens is None: max_tokens = self.reward_max_tokens
        if zeta is None: zeta = self.reward_zeta
        if word_leak_threshold is None: word_leak_threshold = self.reward_word_leak_threshold
        if length_cap is None: length_cap = self.reward_length_cap
        device = self.belief_device
        candidate_words = self.topic_codes[self.topic]

        speaker_idx = self.agent_to_idx[speaker_name]
        true_word_idx = candidate_words.index(self.code)

        rewards = []
        components = []

        for clue in clues:
            pairs = [(clue, word) for word in candidate_words]
            scores = self.backend.batch_score(pairs)
            scores = (scores - scores.mean()) / (scores.std() + 1e-6)
            
            scores = torch.tensor(scores, dtype=torch.float32, device=device)
            log_q = torch.log(self.word_belief + 1e-12) + lmb * scores
            q_new = torch.softmax(log_q, dim=0)

            true_score = scores[true_word_idx]

            baseline = scores.mean()
            suspicion_delta = baseline - true_score

            log_p = torch.log(self.player_belief + 1e-12)
            log_p[speaker_idx] = log_p[speaker_idx] + eta * suspicion_delta
            p_new = torch.softmax(log_p, dim=0)

            self_suspicion = p_new[speaker_idx] - self.player_belief[speaker_idx]
            word_leak = q_new[true_word_idx] - self.word_belief[true_word_idx]
            word_leak_penalty = torch.clamp(word_leak - word_leak_threshold, min=0.0)
            tokenized_clue = self.backend.tokenizer(clue)
            token_number = len(tokenized_clue["input_ids"])

            over_by = max(token_number - max_tokens, 0)
            length_penalty_raw = math.exp(zeta*over_by) - 1
            length_penalty = min(length_penalty_raw, length_cap)
            length_cap_hit = length_penalty_raw > length_cap

            self_suspicion_term = -alpha * self_suspicion
            word_leak_term = -gamma * word_leak_penalty
            length_term = -length_penalty

            clue_reward = self_suspicion_term + word_leak_term + length_term

            rewards.append(clue_reward)
            components.append({
                "self_suspicion": float(self_suspicion),
                "self_suspicion_term": float(self_suspicion_term),
                "word_leak": float(word_leak),
                "word_leak_penalty": float(word_leak_penalty),
                "word_leak_term": float(word_leak_term),
                "token_number": int(token_number),
                "over_by": int(over_by),
                "length_penalty": float(length_penalty),
                "length_penalty_raw": float(length_penalty_raw),
                "length_cap_hit": bool(length_cap_hit),
                "length_term": float(length_term),
                "total": float(clue_reward),
            })

        return rewards, components

    def update_belief(self, speaker_name, clue):
        device = self.word_belief.device
        candidate_words = self.topic_codes[self.topic]

        speaker_idx = self.agent_to_idx[speaker_name]
        true_idx = candidate_words.index(self.code)  # topic-local index

        pairs = [(clue, word) for word in candidate_words]
        scores = self.backend.batch_score(pairs)
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
        scores = (scores - scores.mean()) / (scores.std() + 1e-6)

        log_q = torch.log(self.word_belief + 1e-12) + self.reward_lmb * scores
        q_new = torch.softmax(log_q, dim=0)

        true_score = scores[true_idx]
        baseline = scores.mean()
        suspicion_delta = baseline - true_score

        log_p = torch.log(self.player_belief + 1e-12).clone()
        log_p[speaker_idx] = log_p[speaker_idx] + self.reward_eta * suspicion_delta
        p_new = torch.softmax(log_p, dim=0)

        if speaker_name != self.chameleon_name:
            self.word_belief = q_new.detach()

        self.player_belief = p_new.detach()

    def step(self, player_name: str, action: str) -> TimeStep:
        if not self._initialized:
            self.reset()

        assert player_name == self.get_next_player(), (
            f"Wrong player! It is {self.get_next_player()} turn."
        )

        if self._current_phase == "give clues":
            message = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
            )

            self.message_pool.append_message(message)

            self.update_belief(speaker_name=player_name, clue=action)

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
                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=rewards,
                    terminal=terminal,
                )
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

                    timestep = TimeStep(
                        observation=self.get_observation(),
                        reward=rewards,
                        terminal=terminal,
                        chameleon_won=True,
                        win_method="chameleon-votes",
                    )
                else:
                    self._moderator_speak(
                        f"The accusation is correct! {self.chameleon_name} is the chameleon! "
                        f"Now {self.chameleon_name} can guess the secret code. "
                        'You should say: I guess the code is "..."'
                    )
                    self._current_phase = "guess"
                    rewards = self.get_zero_rewards()
                    terminal = False

                    timestep = TimeStep(
                        observation=self.get_observation(),
                        reward=rewards,
                        terminal=terminal,
                    )

                self._current_turn += 1

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

                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_rewards(chameleon_win=True),
                    terminal=True,
                    chameleon_won=True,
                    win_method="chameleon-guess",
                )

            else:
                self._moderator_speak(
                    f"{player_name} guessed the code wrong! The secret word is {self.code}. "
                    f"{self.non_chameleon_names} won!"
                )
                rewards = self.get_rewards(chameleon_win=False)

                timestep = TimeStep(
                    observation=self.get_observation(),
                    reward=self.get_rewards(chameleon_win=False),
                    terminal=True,
                    chameleon_won=False,
                    win_method="non-chameleon",
                )

        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        if self.is_terminal():
            timestep.terminal = True

        return timestep

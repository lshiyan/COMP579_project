import logging
import re
import uuid
from abc import abstractmethod
from typing import List, Union, Sequence, Optional

import torch
import torch.nn as nn
from tenacity import RetryError

from .backends import IntelligenceBackend, load_backend, TransformersLlamaChat
from .config import AgentConfig, BackendConfig, Configurable
from .message import SYSTEM_NAME, Message

SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):
    @abstractmethod
    def __init__(
        self, name: str, role_desc: str, global_prompt: str = None, *args, **kwargs
    ):
        super().__init__(
            name=name, role_desc=role_desc, global_prompt=global_prompt, **kwargs
        )
        self.name = name
        self.role_desc = role_desc
        self.global_prompt = global_prompt


class Player(Agent):
    VALID_HIDDEN_ROLES = {"chameleon", "non_chameleon"}

    def __init__(
        self,
        name: str,
        role_desc: str,
        backend: Union[BackendConfig, IntelligenceBackend],
        clue_number: int = 3,
        global_prompt: str = None,
        embedding_size: Optional[int] = None,
        belief_state_size: Optional[int] = None,
        hidden_role: Optional[str] = None,
        shared_player_belief_head=None,
        shared_word_belief_head=None,
        shared_belief_updater=None,
        shared_speaker_embedding=None,
        shared_role_embedding=None,
        **kwargs,
    ):
        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(
                f"backend must be a BackendConfig or an IntelligenceBackend, but got {type(backend)}"
            )

        assert (
            name != SYSTEM_NAME
        ), f"Player name cannot be {SYSTEM_NAME}, which is reserved for the system."

        super().__init__(
            name=name,
            role_desc=role_desc,
            backend=backend_config,
            global_prompt=global_prompt,
            **kwargs,
        )

        self.backend = backend
        self.embedding_size = embedding_size
        self.belief_state_size = belief_state_size
        self.clue_number = clue_number

        if self.embedding_size is None:
            raise ValueError("embedding_size must be provided.")
        if self.belief_state_size is None:
            raise ValueError("belief_state_size must be provided.")

        self.agents = []
        self.words = []

        self.agent_to_idx = {}
        self.word_to_idx = {}

        self.hidden_role: Optional[str] = None
        self.beliefs: Optional[torch.Tensor] = None
        self.self_idx: Optional[int] = None
        self.secret_word: Optional[str] = None

        # Shared heads / modules owned by environment
        self.shared_player_belief_head = shared_player_belief_head
        self.shared_word_belief_head = shared_word_belief_head
        self.shared_belief_updater = shared_belief_updater
        self.shared_speaker_embedding = shared_speaker_embedding
        self.shared_role_embedding = shared_role_embedding

        # Per-player recurrent belief state
        self.belief_state: Optional[torch.Tensor] = None

        self._refresh_index_maps()

        if hidden_role is not None:
            self.set_hidden_role(hidden_role)

    def _refresh_index_maps(self):
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.self_idx = self.agent_to_idx.get(self.name, None)

    def set_agents(self, agents: List[str]):
        self.agents = [agent for agent in agents]
        self._refresh_index_maps()

    def set_words(self, words: List[str]):
        self.words = [word for word in words]
        self._refresh_index_maps()

    def set_shared_belief_heads(self, player_head, word_head):
        self.shared_player_belief_head = player_head
        self.shared_word_belief_head = word_head

    def set_shared_belief_modules(
        self,
        belief_updater,
        speaker_embedding,
        role_embedding,
    ):
        self.shared_belief_updater = belief_updater
        self.shared_speaker_embedding = speaker_embedding
        self.shared_role_embedding = role_embedding

    def set_hidden_role(self, hidden_role: str, agents: List[str], words: List[str]):
        if hidden_role not in self.VALID_HIDDEN_ROLES:
            raise ValueError(
                f"hidden_role must be one of {self.VALID_HIDDEN_ROLES}, got {hidden_role}"
            )

        self.hidden_role = hidden_role
        self.set_agents(agents)
        self.set_words(words)

        if hidden_role == "chameleon":
            if len(self.words) == 0:
                raise ValueError("words must be set before assigning role 'chameleon'.")
            if self.shared_word_belief_head is None:
                raise ValueError("shared_word_belief_head has not been set.")

        elif hidden_role == "non_chameleon":
            if len(self.agents) == 0:
                raise ValueError("agents must be set before assigning role 'non_chameleon'.")
            if self.self_idx is None:
                raise ValueError(
                    f"Player {self.name} must appear in agents before assigning role 'non_chameleon'."
                )
            if self.shared_player_belief_head is None:
                raise ValueError("shared_player_belief_head has not been set.")

        if self.shared_belief_updater is None:
            raise ValueError("shared_belief_updater has not been set.")
        if self.shared_speaker_embedding is None:
            raise ValueError("shared_speaker_embedding has not been set.")
        if self.shared_role_embedding is None:
            raise ValueError("shared_role_embedding has not been set.")

        self.reset_beliefs()

    def _get_device(self) -> torch.device:
        modules = [
            self.shared_belief_updater,
            self.shared_player_belief_head,
            self.shared_word_belief_head,
            self.shared_speaker_embedding,
            self.shared_role_embedding,
        ]
        for module in modules:
            if module is not None:
                try:
                    return next(module.parameters()).device
                except StopIteration:
                    pass
        return torch.device("cpu")

    def _get_role_id(self) -> int:
        if self.hidden_role == "non_chameleon":
            return 0
        if self.hidden_role == "chameleon":
            return 1
        raise ValueError("Player hidden role has not been initialized.")

    def reset_beliefs(self):
        if self.hidden_role is None:
            self.beliefs = None
            self.belief_state = None
            return

        device = self._get_device()

        # Initialize recurrent belief state
        self.belief_state = torch.zeros(
            1, self.belief_state_size, dtype=torch.float32, device=device
        )

        if self.hidden_role == "chameleon":
            num_words = len(self.words)
            self.beliefs = torch.full(
                (num_words,),
                1.0 / num_words,
                dtype=torch.float32,
                device=device,
            )

        elif self.hidden_role == "non_chameleon":
            num_players = len(self.agents)
            self.beliefs = torch.zeros(num_players, dtype=torch.float32, device=device)
            valid_indices = [i for i in range(num_players) if i != self.self_idx]
            if len(valid_indices) == 0:
                raise ValueError("Need at least one other player for non-chameleon beliefs.")
            self.beliefs[valid_indices] = 1.0 / len(valid_indices)

    def get_belief_logits(self, hidden_state):
        if self.hidden_role is None:
            raise ValueError("Player hidden role has not been initialized.")

        if self.hidden_role == "non_chameleon":
            logits = self.shared_player_belief_head(hidden_state)
            logits = logits[..., :len(self.agents)]

            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)

            logits = logits.clone()
            logits[self.self_idx] = float("-inf")
            return logits

        elif self.hidden_role == "chameleon":
            logits = self.shared_word_belief_head(hidden_state)
            logits = logits[..., :len(self.words)]

            if logits.dim() == 2 and logits.shape[0] == 1:
                logits = logits.squeeze(0)

            return logits

        raise ValueError(f"Unknown hidden_role: {self.hidden_role}")

    def get_belief(self, hidden_state):
        logits = self.get_belief_logits(hidden_state)
        self.beliefs = torch.softmax(logits, dim=-1)
        return self.beliefs

    def update_belief_state(
        self,
        message_embedding: torch.Tensor,
        speaker_name: str,
    ):
        """
        Update this player's recurrent belief state from a new observed message embedding.

        Args:
            message_embedding: Tensor of shape (embedding_size,) or (1, embedding_size)
            speaker_name: name of the player who produced the message
        """
        if self.hidden_role is None:
            raise ValueError("Player hidden role has not been initialized.")
        if self.shared_belief_updater is None:
            raise ValueError("shared_belief_updater has not been set.")
        if speaker_name not in self.agent_to_idx:
            raise ValueError(f"Unknown speaker_name: {speaker_name}")

        device = self._get_device()

        if message_embedding.dim() == 1:
            message_embedding = message_embedding.unsqueeze(0)
        elif message_embedding.dim() != 2 or message_embedding.shape[0] != 1:
            raise ValueError(
                f"message_embedding must have shape (embedding_size,) or (1, embedding_size), "
                f"got {tuple(message_embedding.shape)}"
            )

        message_embedding = message_embedding.to(device)

        if message_embedding.shape[-1] != self.embedding_size:
            raise ValueError(
                f"Expected message embedding dim {self.embedding_size}, "
                f"got {message_embedding.shape[-1]}"
            )

        if self.belief_state is None:
            self.belief_state = torch.zeros(
                1, self.belief_state_size, dtype=torch.float32, device=device
            )

        speaker_idx = torch.tensor(
            [self.agent_to_idx[speaker_name]], dtype=torch.long, device=device
        )
        role_idx = torch.tensor(
            [self._get_role_id()], dtype=torch.long, device=device
        )

        speaker_emb = self.shared_speaker_embedding(speaker_idx)
        role_emb = self.shared_role_embedding(role_idx)

        updater_input = torch.cat(
            [message_embedding, speaker_emb, role_emb],
            dim=-1,
        )

        self.belief_state = self.shared_belief_updater(
            updater_input,
            self.belief_state,
        )

        self.get_belief(self.belief_state)
        return self.belief_state, self.beliefs

    def save_beliefs(self):
        beliefs = self.beliefs.detach().clone()
        belief_state = self.belief_state.detach().clone()
        
        return beliefs, belief_state

    def set_beliefs(self, beliefs: torch.Tensor, belief_state: torch.Tensor):
        self.beliefs = beliefs
        self.belief_state = belief_state
        
    def give_secret_word(self, word: str):
        if word not in self.word_to_idx:
            raise ValueError(f"Unknown word: {word}")
        self.secret_word = word

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )

    def act(self, observation: List[Message]):
        try:
            return self.backend.query(
                agent_name=self.name,
                role_desc=self.role_desc,
                history_messages=observation,
                global_prompt=self.global_prompt,
                request_msg=None,
            )
        except RetryError as e:
            err_msg = (
                f"Agent {self.name} failed to generate a response. "
                f"Error: {e.last_attempt.exception()}. "
                f"Sending signal to end the conversation."
            )
            logging.warning(err_msg)
            return {
                "response": SIGNAL_END_OF_CONVERSATION + err_msg,
                "new_tokens": None,
                "token_logprobs": None,
                "seq_logprob": None,
                "prompt_input_ids": None,
                "prompt_attention_mask": None,
            }

    def __call__(self, observation: List[Message]):
        return self.act(observation)

    def reset(self):
        self.backend.reset()
        self.reset_beliefs()
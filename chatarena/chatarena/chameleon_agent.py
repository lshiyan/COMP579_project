import logging
import re
import random
import uuid
from abc import abstractmethod
from typing import List, Union, Sequence, Optional

import torch
import torch.nn as nn
from tenacity import RetryError
from contextlib import nullcontext


from .backends import IntelligenceBackend, load_backend, TransformersHuggingFaceChat
from .config import AgentConfig, BackendConfig, Configurable
from .message import SYSTEM_NAME, Message

SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Agent(Configurable):
    @abstractmethod
    def __init__(
        self, name: str, role_desc: str, global_prompt: Optional[str] = None, *args, **kwargs
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
        global_prompt: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(backend, BackendConfig):
            backend_config = backend
            backend = load_backend(backend_config)
        elif isinstance(backend, IntelligenceBackend):
            backend_config = backend.to_config()
        else:
            raise ValueError(
                f"backend must be BackendConfig or IntelligenceBackend, got {type(backend)}"
            )

        assert name != SYSTEM_NAME

        super().__init__(
            name=name,
            role_desc=role_desc,
            backend=backend_config,
            global_prompt=global_prompt,
            **kwargs,
        )

        self.backend = backend
        self.clue_number = clue_number

        self.agents = []
        self.words = []
        self.agent_to_idx = {}
        self.word_to_idx = {}

        self.hidden_role: Optional[str] = None
        self.self_idx: Optional[int] = None

    def _refresh_index_maps(self):
        self.agent_to_idx = {agent: i for i, agent in enumerate(self.agents)}
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.self_idx = self.agent_to_idx.get(self.name, None)

    def set_hidden_role(self, hidden_role: str, agents: List[str], words: List[str]):
        if hidden_role not in self.VALID_HIDDEN_ROLES:
            raise ValueError(f"Invalid hidden_role: {hidden_role}")

        self.hidden_role = hidden_role
        self.agents = list(agents)
        self.words = list(words)
        self._refresh_index_maps()

    def act(self, observation: List[Message]):
        try:
            
            context = self.backend.model.disable_adapter() if self.hidden_role == "chameleon" else nullcontext()
            
            with context:
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

    def vote_from_belief(self, player_belief: torch.Tensor):
        probs = player_belief.clone()

        if self.self_idx is not None:
            probs[self.self_idx] = 0.0

        probs = probs / probs.sum()
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        return self.agents[sampled_idx]

    def guess_from_belief(self, word_belief: torch.Tensor):
        sampled_idx = torch.multinomial(word_belief, num_samples=1).item()
        return self.words[sampled_idx]

    def __call__(self, observation: List[Message]):
        return self.act(observation)

    def reset(self):
        self.backend.reset()

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config(),
            global_prompt=self.global_prompt,
        )
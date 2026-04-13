import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential
from anthropic import APIStatusError
from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend

try:
    import anthropic
except ImportError:
    is_anthropic_available = False
else:
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_api_key is None:
        is_anthropic_available = False
    else:
        is_anthropic_available = True

DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "claude-sonnet-4-20250514"


@register_backend
class Claude(IntelligenceBackend):
    """Interface to the Claude models offered by Anthropic."""

    stateful = False
    type_name = "claude"

    def __init__(
        self, max_tokens: int = DEFAULT_MAX_TOKENS, model: str = DEFAULT_MODEL, **kwargs
    ):
        assert (
            is_anthropic_available
        ), "anthropic package is not installed or the API key is not set"
        super().__init__(max_tokens=max_tokens, model=model, **kwargs)

        self.max_tokens = max_tokens
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(min=5, max=60),
        retry=retry_if_exception(lambda e: isinstance(e, APIStatusError) and e.status_code in {529, 529, 503, 502, 500}),
    )    
    def _get_response(self, system_prompt: str, messages: list) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text.strip()

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        # Build system prompt
        system_parts = []
        if global_prompt:
            system_parts.append(global_prompt)
        system_parts.append(role_desc)
        system_prompt = "\n\n".join(system_parts)

        # Convert history into alternating user/assistant turns
        # Messages API requires strictly alternating roles
        messages = []
        pending_user_parts = []

        def flush_user(parts):
            if parts:
                messages.append({"role": "user", "content": "\n".join(parts)})
            return []

        for message in history_messages:
            if message.agent_name == agent_name:
                pending_user_parts = flush_user(pending_user_parts)
                messages.append({"role": "assistant", "content": message.content})
            else:
                pending_user_parts.append(f"[{message.agent_name}]: {message.content}")

        flush_user(pending_user_parts)

        # Add the final request message as the last user turn
        final_parts = []
        if request_msg:
            final_parts.append(f"[{SYSTEM}]: {request_msg.content}")
        if final_parts:
            messages.append({"role": "user", "content": "\n".join(final_parts)})
        elif not messages or messages[-1]["role"] == "assistant":
            # Messages API requires the last message to be from the user
            messages.append({"role": "user", "content": "Please continue."})

        response = self._get_response(system_prompt, messages, *args, **kwargs)

        # Strip agent name prefix if the model echoes it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()

        return response
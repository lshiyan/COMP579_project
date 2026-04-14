import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend

try:
    from google import genai
    from google.genai import types
except ImportError:
    is_gemini_available = False
else:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key is None:
        is_gemini_available = False
    else:
        _client = genai.Client(api_key=gemini_api_key)
        is_gemini_available = True

DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gemini-2.5-flash"


@register_backend
class Gemini(IntelligenceBackend):
    """Interface to the Gemini models using the new google-genai SDK."""

    stateful = False
    type_name = "gemini"

    def __init__(
        self, max_tokens: int = DEFAULT_MAX_TOKENS, model: str = DEFAULT_MODEL, **kwargs
    ):
        assert (
            is_gemini_available
        ), "google-genai package is not installed or GEMINI_API_KEY is not set"
        super().__init__(max_tokens=max_tokens, model=model, **kwargs)

        self.max_tokens = max_tokens
        self.model = model
        self.client = _client

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, system_prompt: str, history: list, last_user_msg: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt or None,
            max_output_tokens=self.max_tokens,
        )

        # Append the final user turn to history for the API call
        contents = history + [
            types.Content(role="user", parts=[types.Part(text=last_user_msg)])
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text.strip()

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

        # Convert history into new SDK Content objects
        # Gemini requires strictly alternating user/model turns
        contents = []
        pending_user_parts = []

        def flush_user(parts):
            if parts:
                contents.append(
                    types.Content(role="user", parts=[types.Part(text="\n".join(parts))])
                )
            return []

        for message in history_messages:
            if message.agent_name == agent_name:
                pending_user_parts = flush_user(pending_user_parts)
                contents.append(
                    types.Content(role="model", parts=[types.Part(text=message.content)])
                )
            else:
                pending_user_parts.append(f"[{message.agent_name}]: {message.content}")

        flush_user(pending_user_parts)

        # Build the final user message for this turn
        final_parts = []
        if request_msg:
            final_parts.append(f"[{SYSTEM}]: {request_msg.content}")
        last_user_msg = "\n".join(final_parts) if final_parts else "Please continue."

        response = self._get_response(system_prompt, contents, last_user_msg, *args, **kwargs)

        # Strip agent name prefix if the model echoes it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()

        return response
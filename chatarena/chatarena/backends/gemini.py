import os
import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend

try:
    import google.generativeai as genai
except ImportError:
    is_gemini_available = False
    # logging.warning("google-generativeai package is not installed")
else:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key is None:
        # logging.warning("Gemini API key is not set. Please set the environment variable GEMINI_API_KEY")
        is_gemini_available = False
    else:
        genai.configure(api_key=gemini_api_key)
        is_gemini_available = True

DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "gemini-1.5-flash"


@register_backend
class Gemini(IntelligenceBackend):
    """Interface to the Gemini models offered by Google."""

    stateful = False
    type_name = "gemini"

    def __init__(
        self, max_tokens: int = DEFAULT_MAX_TOKENS, model: str = DEFAULT_MODEL, **kwargs
    ):
        assert (
            is_gemini_available
        ), "google-generativeai package is not installed or the API key is not set"
        super().__init__(max_tokens=max_tokens, model=model, **kwargs)

        self.max_tokens = max_tokens
        self.model = model

        self._client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(max_output_tokens=self.max_tokens),
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, system_prompt: str, history: list, last_user_msg: str) -> str:
        # Gemini supports a system_instruction at the model level, but we pass it
        # as part of the chat history for compatibility with the base client.
        client = genai.GenerativeModel(
            model_name=self.model,
            generation_config=genai.GenerationConfig(max_output_tokens=self.max_tokens),
            system_instruction=system_prompt if system_prompt else None,
        )

        chat = client.start_chat(history=history)
        response = chat.send_message(last_user_msg)
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
        """
        Format the input and call the Gemini API.

        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            history_messages: the history of the conversation, or the observation for the agent
            request_msg: the request from the system to guide the agent's next response
        """
        # Build the system prompt from global_prompt and role_desc
        system_parts = []
        if global_prompt:
            system_parts.append(global_prompt)
        system_parts.append(role_desc)
        system_prompt = "\n\n".join(system_parts)

        # Convert history_messages into Gemini's chat history format.
        # Gemini uses alternating "user" / "model" roles.
        # Messages from agent_name -> "model", everything else -> "user"
        gemini_history = []
        pending_user_parts = []

        def flush_user(parts):
            if parts:
                gemini_history.append({"role": "user", "parts": ["\n".join(parts)]})
            return []

        for message in history_messages:
            if message.agent_name == agent_name:
                # Flush any accumulated user messages first
                pending_user_parts = flush_user(pending_user_parts)
                gemini_history.append(
                    {"role": "model", "parts": [message.content]}
                )
            else:
                pending_user_parts.append(f"[{message.agent_name}]: {message.content}")

        # Flush remaining user messages before the final turn
        pending_user_parts = flush_user(pending_user_parts)

        # Build the final user message for this turn
        final_user_parts = []
        if request_msg:
            final_user_parts.append(f"[{SYSTEM}]: {request_msg.content}")

        # If there's nothing to send as a final user message, use a minimal prompt
        last_user_msg = "\n".join(final_user_parts) if final_user_parts else "Please continue."

        response = self._get_response(system_prompt, gemini_history, last_user_msg, *args, **kwargs)

        # Remove the agent name if the response starts with it
        response = re.sub(rf"^\s*\[{agent_name}]:?", "", response).strip()

        return response
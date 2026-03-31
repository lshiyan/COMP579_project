import re
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME, Message
from .base import IntelligenceBackend, register_backend

try:
    import openai
except ImportError:
    is_openai_available = False
else:
    is_openai_available = True

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434/v1"

END_OF_MESSAGE = "<EOS>"  # End of message token
STOP = ("<|endoftext|>", END_OF_MESSAGE)
BASE_PROMPT = f"The messages always end with the token {END_OF_MESSAGE}."


@register_backend
class OllamaChat(IntelligenceBackend):
    """Backend for locally-hosted models via Ollama's OpenAI-compatible API."""

    stateful = False
    type_name = "ollama"

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        merge_other_agents_as_one_user: bool = True,
        **kwargs,
    ):
        assert is_openai_available, "openai package is not installed"
        super().__init__(
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            base_url=base_url,
            merge_other_agents_as_one_user=merge_other_agents_as_one_user,
            **kwargs,
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.base_url = base_url
        self.merge_other_agent_as_user = merge_other_agents_as_one_user

        # Create an OpenAI client pointed at the Ollama server
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=list(STOP),
        )

        response = completion.choices[0].message.content
        response = response.strip()
        return response

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
        if global_prompt:
            system_prompt = (
                f"You are a helpful assistant.\n{global_prompt.strip()}\n{BASE_PROMPT}\n\n"
                f"Your name is {agent_name}.\n\nYour role:{role_desc}"
            )
        else:
            system_prompt = (
                f"You are a helpful assistant. Your name is {agent_name}.\n\n"
                f"Your role:{role_desc}\n\n{BASE_PROMPT}"
            )

        all_messages = [(SYSTEM_NAME, system_prompt)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))

        if request_msg:
            all_messages.append((SYSTEM_NAME, request_msg.content))
        else:
            all_messages.append(
                (SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}")
            )

        messages = []
        for i, msg in enumerate(all_messages):
            if i == 0:
                assert msg[0] == SYSTEM_NAME
                messages.append({"role": "system", "content": msg[1]})
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    if messages[-1]["role"] == "user":
                        if self.merge_other_agent_as_user:
                            messages[-1]["content"] = (
                                f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
                            )
                        else:
                            messages.append(
                                {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                            )
                    elif messages[-1]["role"] == "assistant":
                        # Start a new user turn rather than merging into the agent's own assistant turn
                        messages.append(
                            {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                        )
                    elif messages[-1]["role"] == "system":
                        messages.append(
                            {"role": "user", "content": f"[{msg[0]}]: {msg[1]}"}
                        )
                    else:
                        raise ValueError(f"Invalid role: {messages[-1]['role']}")

        response = self._get_response(messages, *args, **kwargs)

        # Clean up the response
        response = re.sub(rf"^\s*\[.*]:", "", response).strip()
        response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
        response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response

    async def async_query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        # Fall back to sync for now
        return self.query(
            agent_name, role_desc, history_messages, global_prompt, request_msg,
            *args, **kwargs,
        )

import os
import torch
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_stdout_stderr():
    try:
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        is_transformers_available = False
    else:
        is_transformers_available = True


@register_backend
class TransformersLlamaChat(IntelligenceBackend):
    """Interface to a Hugging Face LLaMA-style chat model."""

    stateful = False
    type_name = "transformers:llama-chat"

    def __init__(
        self,
        model: str,
        device: int = -1,
        torch_dtype: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        assert is_transformers_available, "Transformers package is not installed"

        # Resolve dtype
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

        # Resolve device / device_map
        if device >= 0 and torch.cuda.is_available():
            # Put the whole model on a specific CUDA device
            device_map = {"": device}
        else:
            device_map = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

        # Some LLaMA tokenizers do not have a pad token by default
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def _to_chat_messages(
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
    ):
        """
        Convert the framework's message format into HF chat-template format.
        """
        messages = []

        system_parts = []
        if global_prompt:
            system_parts.append(global_prompt)
        if role_desc:
            system_parts.append(role_desc)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for msg in history_messages:
            role = "assistant" if msg.agent_name == agent_name else "user"
            content = f"[{msg.agent_name}]: {msg.content}"
            messages.append({"role": role, "content": content})

        if request_msg is not None:
            messages.append({"role": "user", "content": f"[{SYSTEM}]: {request_msg.content}"})

        return messages

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages) -> str:
        """
        Run generation using the tokenizer's chat template.
        """
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move tokenized inputs to the model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode newly generated tokens
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, prompt_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
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
        messages = self._to_chat_messages(
            agent_name=agent_name,
            role_desc=role_desc,
            history_messages=history_messages,
            global_prompt=global_prompt,
            request_msg=request_msg,
        )
        return self._get_response(messages)
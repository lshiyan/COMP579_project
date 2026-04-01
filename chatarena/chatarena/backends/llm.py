import os
import torch
import torch.nn.functional as F
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List, Tuple, Union, Dict, Any, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

@register_backend
class TransformersLlamaChat(IntelligenceBackend):
    """Interface to a Hugging Face LLaMA-style chat model with a SentenceTransformer encoder."""

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
        latent_pooling: str = "mean",
        normalize_latent: bool = False,
        sentence_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_sentence_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            latent_pooling=latent_pooling,
            normalize_latent=normalize_latent,
            sentence_encoder_model=sentence_encoder_model,
            normalize_sentence_embeddings=normalize_sentence_embeddings,
            **kwargs,
        )
        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.latent_pooling = latent_pooling
        self.normalize_latent = normalize_latent

        self.sentence_encoder_model_name = sentence_encoder_model
        self.normalize_sentence_embeddings = normalize_sentence_embeddings

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

        if device >= 0 and torch.cuda.is_available():
            device_map = {"": device}
            sentence_encoder_device = f"cuda:{device}"
        else:
            device_map = "cpu"
            sentence_encoder_device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sentence_encoder = SentenceTransformer(
            self.sentence_encoder_model_name,
            device=sentence_encoder_device,
        )
        self.sentence_embedding_size = self.sentence_encoder.get_sentence_embedding_dimension()

    @staticmethod
    def _to_chat_messages(
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
    ):
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

    def _tokenize_messages(
        self,
        messages,
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    def _pool_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)
        returns: (batch, hidden_dim)
        """
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            pooled = summed / counts

        elif pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            pooled = hidden_states[
                torch.arange(hidden_states.size(0), device=hidden_states.device),
                lengths,
            ]

        elif pooling == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states.masked_fill(~mask, float("-inf"))
            pooled = masked_hidden.max(dim=1).values

        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")

        if self.normalize_latent:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled

    @torch.no_grad()
    def _get_latent_state(
        self,
        messages,
        pooling: str = None,
        add_generation_prompt: bool = True,
        return_token_level: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a latent representation of the full chat history/prompt.

        If return_token_level=False:
            returns pooled latent of shape (batch, hidden_dim)

        If return_token_level=True:
            returns:
                pooled_latent: (batch, hidden_dim)
                token_hidden:  (batch, seq_len, hidden_dim)
        """
        if pooling is None:
            pooling = self.latent_pooling

        inputs = self._tokenize_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
        )

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        token_hidden = outputs.hidden_states[-1]
        pooled = self._pool_hidden(
            hidden_states=token_hidden,
            attention_mask=inputs["attention_mask"],
            pooling=pooling,
        )

        if return_token_level:
            return pooled, token_hidden
        return pooled

    @torch.no_grad()
    def _encode_text(
        self,
        text: Union[str, List[str]],
        convert_to_tensor: bool = True,
        detach_to_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Encode raw text with the SentenceTransformer.

        Args:
            text: a single string or list of strings
            convert_to_tensor: whether to return a torch.Tensor
            detach_to_cpu: if True, move output to CPU

        Returns:
            If text is str: Tensor of shape (D,) or (1, D)
            If text is List[str]: Tensor of shape (N, D)
        """
        embedding = self.sentence_encoder.encode(
            text,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=self.normalize_sentence_embeddings,
            show_progress_bar=False,
        )

        if convert_to_tensor:
            embedding = embedding.float()
            if detach_to_cpu:
                embedding = embedding.detach().cpu()

        return embedding

    @torch.no_grad()
    def get_message_embedding(
        self,
        message_text: str,
        detach_to_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Convenience wrapper for encoding a single speaker-tagged message.
        """
        return self._encode_text(message_text, convert_to_tensor=True, detach_to_cpu=detach_to_cpu)

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages) -> str:
        inputs = self._tokenize_messages(messages, add_generation_prompt=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

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
        return_latent_state: bool = False,
        latent_pooling: str = "mean",
        return_token_level_latent: bool = False,
        detach_to_cpu: bool = True,
        *args,
        **kwargs,
    ) -> Union[str, Tuple[str, torch.Tensor], Tuple[str, torch.Tensor, torch.Tensor]]:
        """
        Default:
            returns response: str

        If return_latent_state=True and return_token_level_latent=False:
            returns (response, latent_state)

        If return_latent_state=True and return_token_level_latent=True:
            returns (response, pooled_latent_state, token_level_hidden_states)

        Note:
            This latent_state is still the LLaMA latent over the prompt/history.
            For recurrent belief updates, use encode_text(...) or get_message_embedding(...).
        """
        messages = self._to_chat_messages(
            agent_name=agent_name,
            role_desc=role_desc,
            history_messages=history_messages,
            global_prompt=global_prompt,
            request_msg=request_msg,
        )

        print(messages)
        
        latent = None
        token_level_latent = None

        if return_latent_state:
            latent_out = self._get_latent_state(
                messages,
                pooling=latent_pooling,
                add_generation_prompt=True,
                return_token_level=return_token_level_latent,
            )

            if return_token_level_latent:
                latent, token_level_latent = latent_out
            else:
                latent = latent_out

            if detach_to_cpu:
                latent = latent.detach().cpu()
                if token_level_latent is not None:
                    token_level_latent = token_level_latent.detach().cpu()

        response = self._get_response(messages)

        if not return_latent_state:
            return response

        if return_token_level_latent:
            return response, latent, token_level_latent

        return response, latent
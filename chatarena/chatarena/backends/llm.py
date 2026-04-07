import os
import torch
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List, Tuple, Union, Dict

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@register_backend
class TransformersLlamaChat(IntelligenceBackend):
    """
    Hugging Face LLaMA-style chat backend for:
    - sampling chat responses
    - returning token-level / sequence-level logprobs
    - encoding messages for belief updates
    """

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
        sentence_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_sentence_embeddings: bool = True,
        lora_cfg: dict = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            sentence_encoder_model=sentence_encoder_model,
            normalize_sentence_embeddings=normalize_sentence_embeddings,
            **kwargs,
        )

        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

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

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            local_files_only=True,
        )
        
        if lora_cfg is not None:
            task_type_str = lora_cfg.pop("task_type", "CAUSAL_LM")
            lora_config = LoraConfig(
                task_type=TaskType[task_type_str],
                **lora_cfg
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()
            self.model.print_trainable_parameters()
        
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.sentence_encoder = SentenceTransformer(
            self.sentence_encoder_model_name,
            device=sentence_encoder_device,
        )
        self.sentence_embedding_size = (
            self.sentence_encoder.get_sentence_embedding_dimension()
        )

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
            messages.append(
                {"role": "user", "content": f"[{SYSTEM}]: {request_msg.content}"}
            )

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

    @torch.no_grad()
    def _encode_text(
        self,
        text: Union[str, List[str]],
        convert_to_tensor: bool = True,
        detach_to_cpu: bool = True,
    ) -> torch.Tensor:
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
        return self._encode_text(
            message_text,
            convert_to_tensor=True,
            detach_to_cpu=detach_to_cpu,
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        inputs = self._tokenize_messages(messages, add_generation_prompt=True)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

        prompt_len = inputs["input_ids"].shape[1]

        sequences = outputs.sequences
        new_tokens = sequences[:, prompt_len:]

        response = self.tokenizer.decode(
            new_tokens[0], skip_special_tokens=True
        ).strip()

        token_logprobs = []
        for t, step_scores in enumerate(outputs.scores):
            step_logprobs = torch.log_softmax(step_scores, dim=-1)
            chosen_tokens = new_tokens[:, t].unsqueeze(-1)
            chosen_logprobs = step_logprobs.gather(-1, chosen_tokens).squeeze(-1)
            token_logprobs.append(chosen_logprobs)

        if len(token_logprobs) == 0:
            token_logprobs = torch.empty(
                (sequences.shape[0], 0),
                dtype=torch.float32,
                device=sequences.device,
            )
        else:
            token_logprobs = torch.stack(token_logprobs, dim=1)

        seq_logprob = token_logprobs.sum(dim=1)

        return {
            "action": response,
            "new_tokens": new_tokens,
            "token_logprobs": token_logprobs,
            "seq_logprob": seq_logprob,
            "prompt_input_ids": inputs["input_ids"],
            "prompt_attention_mask": inputs["attention_mask"],
        }

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        detach_to_cpu: bool = True,
    ):
        messages = self._to_chat_messages(
            agent_name=agent_name,
            role_desc=role_desc,
            history_messages=history_messages,
            global_prompt=global_prompt,
            request_msg=request_msg,
        )

        out = self._get_response(messages)

        if detach_to_cpu:
            out["new_tokens"] = out["new_tokens"].detach().cpu()
            out["token_logprobs"] = out["token_logprobs"].detach().cpu()
            out["seq_logprob"] = out["seq_logprob"].detach().cpu()
            out["prompt_input_ids"] = out["prompt_input_ids"].detach().cpu()
            out["prompt_attention_mask"] = out["prompt_attention_mask"].detach().cpu()

        return out

    def get_ref_model(self) -> AutoModelForCausalLM:
        ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.model.dtype,
            device_map={"": next(self.model.parameters()).device},
            local_files_only=True,
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
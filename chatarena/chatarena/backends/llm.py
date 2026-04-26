import os
import torch
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List, Tuple, Union, Dict

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, TaskType


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@register_backend
class TransformersHuggingFaceChat(IntelligenceBackend):
    """
    Hugging Face chat backend for:
    - sampling chat responses
    - returning token-level / sequence-level logprobs
    - encoding messages for belief updates
    """

    stateful = False
    type_name = "transformers:huggingface-chat"

    def __init__(
        self,
        model: str,
        device: int = -1,
        torch_dtype: str = "auto",
        max_new_tokens: int = 32, # IMPORTANT: Controls how many words can be in the clue.
        temperature: float = 1.0,
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
            normalize_sentence_embeddings=normalize_sentence_embeddings,
            **kwargs,
        )

        self.model_name = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
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
        else:
            device_map = "cpu"

        # Resolve to a local snapshot path first (avoids transformers 5.x
        # hang with local_files_only=True while still using the cache).
        try:
            from huggingface_hub import snapshot_download
            resolved_path = snapshot_download(self.model_name, local_files_only=True)
        except Exception:
            resolved_path = self.model_name  # fallback: let transformers download

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=dtype,
            device_map=device_map,
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

        self.scorer_model_name = "BAAI/bge-reranker-v2-m3"
        self.clue_scorer = AutoModelForSequenceClassification.from_pretrained(self.scorer_model_name)
        self.scorer_tokenizer = AutoTokenizer.from_pretrained(self.scorer_model_name)
        if device >= 0 and torch.cuda.is_available():
            self.clue_scorer = self.clue_scorer.to(f"cuda:{device}")
        self.clue_scorer.eval()

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
    def get_message_embedding(
        self,
        message_text: str,
        detach_to_cpu: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "get_message_embedding was removed in encoder_belief; use score()/batch_score()."
        )

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages):
        inputs = self._tokenize_messages(messages, add_generation_prompt=True)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
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

    def _format(self, text_a: str, text_b: str):
        """
        Apply consistent formatting for clue-word pairs
        """
        return f"clue: {text_a}", f"word: {text_b}"

    def score(self, text_a: str, text_b: str, normalize: bool = False):
        text_a, text_b = self._format(text_a, text_b)

        scorer_device = next(self.clue_scorer.parameters()).device
        inputs = self.scorer_tokenizer(
            text_a,
            text_b,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(scorer_device)

        with torch.no_grad():
            logits = self.clue_scorer(**inputs).logits.squeeze()

        score = logits.item()

        if normalize:
            score = torch.sigmoid(torch.tensor(score)).item()

        return score

    def batch_score(self, pairs, normalize: bool = False):
        """
        pairs: List[(text_a, text_b)]
        """
        formatted_pairs = [self._format(a, b) for a, b in pairs]
        texts_a, texts_b = zip(*formatted_pairs)

        scorer_device = next(self.clue_scorer.parameters()).device
        inputs = self.scorer_tokenizer(
            list(texts_a),
            list(texts_b),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(scorer_device)

        with torch.no_grad():
            logits = self.clue_scorer(**inputs).logits.squeeze(-1)

        scores = logits.cpu()

        if normalize:
            scores = torch.softmax(scores, dim=0)

        return scores
    
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
        try:
            from huggingface_hub import snapshot_download
            resolved_path = snapshot_download(self.model_name, local_files_only=True)
        except Exception:
            resolved_path = self.model_name
        ref_model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=self.model.dtype,
            device_map={"": "cpu"},
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
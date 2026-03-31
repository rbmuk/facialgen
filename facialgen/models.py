from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - torch is optional for the package
    raise ImportError("facialgen.models requires PyTorch.") from exc

try:
    from transformers import GPT2Config, GPT2LMHeadModel
except ImportError:  # pragma: no cover - transformers is optional
    GPT2Config = None  # type: ignore[assignment]
    GPT2LMHeadModel = None  # type: ignore[assignment]


def _require_transformers() -> None:
    if GPT2Config is None or GPT2LMHeadModel is None:
        raise ImportError(
            "facialgen.models requires the Hugging Face transformers package. "
            "Install it with `pip install transformers` in your project environment."
        )


@dataclass
class FacialGenConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    embd_dropout: float | None = None
    attn_dropout: float | None = None
    resid_dropout: float | None = None
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def to_hf_config(self) -> Any:
        _require_transformers()
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.block_size,
            n_ctx=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            resid_pdrop=(
                self.resid_dropout if self.resid_dropout is not None else self.dropout
            ),
            embd_pdrop=(
                self.embd_dropout if self.embd_dropout is not None else self.dropout
            ),
            attn_pdrop=(
                self.attn_dropout if self.attn_dropout is not None else self.dropout
            ),
            layer_norm_epsilon=self.layer_norm_epsilon,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
        )


class FacialGen(nn.Module):
    """
    Thin wrapper around Hugging Face GPT2LMHeadModel for facial-walk token modeling.

    The model expects the same batch format as the current dataloader:
    `input_ids`, `attention_mask`, and `labels` with pad positions set to `-100`.
    """

    def __init__(self, config: FacialGenConfig) -> None:
        super().__init__()
        _require_transformers()
        self.config = config
        self.hf_config = config.to_hf_config()
        self.model = GPT2LMHeadModel(self.hf_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        if top_k is not None:
            generation_kwargs["top_k"] = top_k

        return self.model.generate(**generation_kwargs)

    def save_pretrained(self, save_directory: str) -> None:
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "FacialGen":
        _require_transformers()
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        hf_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        hf_config = hf_model.config
        model.config = FacialGenConfig(
            vocab_size=hf_config.vocab_size,
            block_size=hf_config.n_positions,
            n_layer=hf_config.n_layer,
            n_head=hf_config.n_head,
            n_embd=hf_config.n_embd,
            dropout=hf_config.resid_pdrop,
            embd_dropout=hf_config.embd_pdrop,
            attn_dropout=hf_config.attn_pdrop,
            resid_dropout=hf_config.resid_pdrop,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            pad_token_id=hf_config.pad_token_id,
        )
        model.hf_config = hf_config
        model.model = hf_model
        return model

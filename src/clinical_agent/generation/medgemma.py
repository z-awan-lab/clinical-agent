"""MedGemma generator wrapping HuggingFace transformers.

Lazy-loaded so importing the package is cheap. The model is loaded on
first ``generate`` call; on the H100 with 4-bit quantisation the 27B
text-only variant fits comfortably and inference is reasonable.

Default repo: ``google/medgemma-27b-text-it`` (MedGemma 1; the 1.5
refresh only shipped 4B at time of writing). Override via the
``model_name`` arg.

Authentication: the model is gated. Either log in via
``huggingface-cli login`` or set ``HF_TOKEN`` in the environment.

Compatibility: requires ``transformers >= 4.50.0`` for Gemma 3 support
(MedGemma is built on Gemma 3). Older transformers will raise a
``ValueError`` on the nested config — the same bug fought in Project 1.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

from .base import BaseGenerator, GenerationResult

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class MedGemmaGenerator(BaseGenerator):
    """MedGemma text generator, 4-bit quantised by default.

    Designed to run on a single H100. CPU-only operation is not
    supported (the unquantised model is too large and bitsandbytes
    requires CUDA).
    """

    model_name: ClassVar[str] = "google/medgemma-27b-text-it"

    def __init__(
        self,
        model_name: str = "google/medgemma-27b-text-it",
        device: str = "cuda",
        load_in_4bit: bool = True,
        torch_dtype: str = "bfloat16",
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name  # type: ignore[misc]
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    # ---- lazy load ---------------------------------------------------

    def _ensure_loaded(self) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "MedGemmaGenerator requires transformers, torch, and "
                "bitsandbytes. Install with: pip install -e '.[ml]'"
            ) from exc

        logger.info("loading tokenizer: %s", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, self.torch_dtype),
                bnb_4bit_use_double_quant=True,
            )

        logger.info(
            "loading model: %s (4-bit=%s, dtype=%s)",
            self.model_name,
            self.load_in_4bit,
            self.torch_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            torch_dtype=getattr(torch, self.torch_dtype),
            device_map=self.device,
            token=self.hf_token,
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        return model, tokenizer

    # ---- BaseGenerator -----------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        import torch

        model, tokenizer = self._ensure_loaded()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_count = int(inputs["input_ids"].shape[1])

        # Stop strings are not supported uniformly across transformers
        # versions. Generate and trim post-hoc against the requested
        # stops — keeps the implementation portable.
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0, prompt_token_count:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_token_count = int(new_tokens.shape[0])

        finish_reason = "length"
        if stop:
            cut = _earliest_stop(text, stop)
            if cut is not None:
                text = text[:cut]
                finish_reason = "stop"

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_token_count,
            finish_reason=finish_reason,
            raw={"stop": stop},
        )


def _earliest_stop(text: str, stops: list[str]) -> int | None:
    """Return the earliest stop-string index in ``text``, or None."""
    earliest: int | None = None
    for s in stops:
        i = text.find(s)
        if i == -1:
            continue
        if earliest is None or i < earliest:
            earliest = i
    return earliest

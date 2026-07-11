from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Protocol, runtime_checkable

import anyio
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

from app.llms.models import Model


@runtime_checkable
class _GenerativeModel(Protocol):
    @property
    def device(self) -> torch.device: ...

    def generate(self, **kwargs: object) -> object: ...


@dataclass
class _RuntimeCache:
    lock: Lock = field(default_factory=Lock)
    runtime: tuple[_GenerativeModel, PreTrainedTokenizerBase] | None = None


_cache = _RuntimeCache()


def _get_runtime() -> tuple[_GenerativeModel, PreTrainedTokenizerBase]:
    with _cache.lock:
        if _cache.runtime is None:
            tokenizer = AutoTokenizer.from_pretrained(Model.QWEN3_8B.value)
            model = AutoModelForCausalLM.from_pretrained(
                Model.QWEN3_8B.value,
                device_map="auto",
                dtype=torch.float16,
            )
            if not isinstance(model, _GenerativeModel):
                raise TypeError("Loaded Qwen model does not support generation")
            _cache.runtime = model, tokenizer
        return _cache.runtime


def _next_token(streamer: Iterator[str]) -> str | None:
    try:
        return next(streamer)
    except StopIteration:
        return None


async def stream_qwen3_response(
    user_prompt: str,
    system_prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> AsyncGenerator[str]:
    model, tokenizer = await anyio.to_thread.run_sync(_get_runtime)
    prompt = tokenizer.apply_chat_template(
        [
            {"content": system_prompt, "role": "system"},
            {"content": user_prompt, "role": "user"},
        ],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_tokens,
                streamer=streamer,
                temperature=temperature,
                top_k=20,
                top_p=top_p,
            )
        except BaseException as error:
            errors.append(error)
            streamer.on_finalized_text("", stream_end=True)

    worker = Thread(target=generate, daemon=True)
    worker.start()
    while (token := await anyio.to_thread.run_sync(_next_token, streamer)) is not None:
        yield token
    await anyio.to_thread.run_sync(worker.join)
    if errors:
        raise errors[0]

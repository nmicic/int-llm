#!/usr/bin/env python3
"""Thin local tokenizer bridge for llama_int text prompts.

This keeps text encode/decode out of the integer-only C inference core while
still allowing real prompt/response testing with local Hugging Face tokenizers.
"""

from __future__ import annotations

import sys
from pathlib import Path


def load_tokenizer(model_dir: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "transformers is required for --prompt text mode "
            "(expected in ./venv/bin/python)"
        ) from exc

    return AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_tokens(path: str) -> list[int]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    return [int(tok) for tok in raw.replace(",", " ").split()]


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)
    return tokenizer.encode(prompt, add_special_tokens=True)


def decode_tokens(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in {"encode", "decode"}:
        print(
            "Usage: hf_tokenizer_bridge.py <encode|decode> <model_dir> <input_file>",
            file=sys.stderr,
        )
        return 1

    mode, model_dir, input_file = argv[1], argv[2], argv[3]
    tokenizer = load_tokenizer(model_dir)

    if mode == "encode":
        prompt = read_text(input_file)
        tokens = encode_prompt(tokenizer, prompt)
        print(" ".join(str(tok) for tok in tokens))
        return 0

    tokens = read_tokens(input_file)
    print(decode_tokens(tokenizer, tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

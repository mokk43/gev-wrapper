from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer

from decider_service.deployment import DeploymentValidationError

_QWEN35_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_backend_tokenizer(metadata_directory: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        metadata_directory,
        local_files_only=True,
    )
    if "pretokenize_regex" not in tokenizer.init_kwargs:
        return tokenizer

    pattern = tokenizer.init_kwargs["pretokenize_regex"]
    pretokenizer = tokenizer.backend_tokenizer.pre_tokenizer
    if (
        type(tokenizer).__name__ != "Qwen2Tokenizer"
        or pattern != _QWEN35_PRETOKENIZE_REGEX
        or not isinstance(pretokenizer, pre_tokenizers.Sequence)
    ):
        raise DeploymentValidationError(
            "tokenizer_config.json declares an unsupported native "
            "pretokenization profile"
        )
    state = json.loads(pretokenizer.__getstate__())
    components = state["pretokenizers"]
    if (
        len(components) != 2
        or components[0]["type"] != "Split"
        or components[0]["behavior"] != "Isolated"
        or components[0]["invert"]
        or components[1]["type"] != "ByteLevel"
        or components[1]["use_regex"]
    ):
        raise DeploymentValidationError(
            "native Qwen3.5 metadata requires a split byte-level tokenizer"
        )

    # The pinned Qwen2 loader ignores this metadata override. llama.cpp's
    # native Qwen3.5 BPE splits raw text, without the loader's NFC composition.
    pretokenizer[0] = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    tokenizer.backend_tokenizer.normalizer = None
    return tokenizer

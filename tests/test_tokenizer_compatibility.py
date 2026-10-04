from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen2Tokenizer

from decider_service.deployment import DeploymentValidationError

NATIVE_QWEN35_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@pytest.fixture
def qwen_metadata(tmp_path: Path) -> Path:
    vocabulary = {
        character: index for index, character in enumerate(sorted(ByteLevel.alphabet()))
    }
    merges: list[tuple[str, str]] = []
    merged = "_"
    for character in "visible":
        merges.append((merged, character))
        merged += character
        vocabulary[merged] = len(vocabulary)
    # Qwen2 forwards BPE merge tuples despite annotating merges as list[str].
    tokenizer = Qwen2Tokenizer(vocab=vocabulary, merges=merges)  # type: ignore[arg-type]
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


def write_tokenizer_config(directory: Path, **overrides: object) -> None:
    path = directory / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config.update(overrides)
    path.write_text(json.dumps(config))


def load_tokenizer(directory: Path) -> Any:
    from decider_service.tokenizer import load_backend_tokenizer

    return load_backend_tokenizer(directory)


def test_declared_native_profile_preserves_mark_boundary_merges(
    qwen_metadata: Path,
) -> None:
    write_tokenizer_config(qwen_metadata, pretokenize_regex=NATIVE_QWEN35_REGEX)
    tokenizer = load_tokenizer(qwen_metadata)
    content = "े_visible"

    token_ids = tokenizer.encode(content, add_special_tokens=False)

    assert token_ids[-1] == tokenizer.convert_tokens_to_ids("_visible")
    assert tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) == content


def test_declared_native_profile_preserves_decomposed_unicode(
    qwen_metadata: Path,
) -> None:
    write_tokenizer_config(qwen_metadata, pretokenize_regex=NATIVE_QWEN35_REGEX)
    tokenizer = load_tokenizer(qwen_metadata)
    content = "e\u0301_visible"

    token_ids = tokenizer.encode(content, add_special_tokens=False)

    assert tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) == content


def test_undeclared_profile_preserves_existing_normalizer(qwen_metadata: Path) -> None:
    tokenizer = load_tokenizer(qwen_metadata)
    token_ids = tokenizer.encode("e\u0301", add_special_tokens=False)

    assert tokenizer.decode(token_ids, clean_up_tokenization_spaces=False) == "é"


@pytest.mark.parametrize("declared_regex", [None, "", "[", "\\p{L}+", 7, [], {}])
def test_rejects_malformed_or_unsupported_declared_profile(
    qwen_metadata: Path,
    declared_regex: object,
) -> None:
    write_tokenizer_config(qwen_metadata, pretokenize_regex=declared_regex)

    with pytest.raises(DeploymentValidationError):
        load_tokenizer(qwen_metadata)


def test_rejects_native_profile_on_unrecognized_tokenizer_class(
    qwen_metadata: Path,
) -> None:
    write_tokenizer_config(
        qwen_metadata,
        pretokenize_regex=NATIVE_QWEN35_REGEX,
        tokenizer_class="PreTrainedTokenizerFast",
    )

    with pytest.raises(DeploymentValidationError):
        load_tokenizer(qwen_metadata)

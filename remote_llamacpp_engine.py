from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import requests
import torch
from transformers import AutoTokenizer

from decider.prompt import MAX_OPTIONS, letter_ids
from decider.temperature import scaled_softmax, slot_temperatures


class RemoteLlamaCppEngine:
    """
    Decider Engine implementation backed by an existing llama.cpp server.

    llama.cpp endpoint:
        POST {base_url}/completion

    This engine intentionally supports one Decider answer slot per request.
    That corresponds naturally to:

        system_one(..., independent=True)

    including isolated Score levels.

    It does NOT load the GGUF file locally.
    """

    def __init__(
        self,
        base_url: str,
        tokenizer_dir: str,
        *,
        timeout: float = 60.0,
        n_probs: int | None = None,
        exact_n_probs: bool = False,
        cache_prompt: bool = False,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_prompt = cache_prompt

        self.session = session or requests.Session()

        # Only tokenizer metadata is needed locally.
        # No GGUF weights are loaded.
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)

        # Token IDs for A, B, C ... used by Decider.
        self.letters = np.asarray(
            letter_ids(self.tok),
            dtype=np.int64,
        )

        # Decider expects engine.m.tok.
        self.m = SimpleNamespace(tok=self.tok)

        self.dev = "remote-llama.cpp"
        self.stats = {
            "forwards": 0,
            "requests": 0,
        }

        if exact_n_probs:
            self.n_probs = self._get_vocab_size()
        else:
            self.n_probs = n_probs or 256

        self.cfg = {
            "backend": "remote-llama.cpp",
            "base_url": self.base_url,
            "n_probs": self.n_probs,
        }

    # ------------------------------------------------------------------
    # llama.cpp metadata
    # ------------------------------------------------------------------

    def _get_vocab_size(self) -> int:
        """
        GET /v1/models returns, on current llama.cpp:

        {
          "data": [{
            "meta": {
              "n_vocab": 248320,
              ...
            }
          }]
        }
        """
        r = self.session.get(
            f"{self.base_url}/v1/models",
            timeout=self.timeout,
        )
        r.raise_for_status()

        body = r.json()

        try:
            return int(body["data"][0]["meta"]["n_vocab"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Cannot determine llama.cpp vocabulary size from "
                f"{self.base_url}/v1/models; "
                "set n_probs explicitly instead."
            ) from exc

    # ------------------------------------------------------------------
    # llama.cpp /completion
    # ------------------------------------------------------------------

    def _completion_logits(
        self,
        token_ids: list[int],
        label_token_ids: list[int],
    ) -> np.ndarray:
        """
        Evaluate one Decider row.

        `token_ids` must end at the Decider answer slot:

            Answer: (

        llama.cpp predicts the next token, so the probabilities returned for
        prediction #0 correspond to the logits Decider wants at that slot.
        """

        payload = {
            # IMPORTANT:
            # raw IDs preserve the exact tokenization Decider produced.
            "prompt": [int(x) for x in token_ids],

            # We need exactly the distribution after the final prompt token.
            "n_predict": 1,

            # Current llama.cpp behavior:
            #
            # temperature < 0 => greedy token selection while n_probs is
            # computed by plain softmax over logits, without the sampling
            # transformations.
            #
            # Decider applies its own fitted temperature later.
            "temperature": -1.0,

            "n_probs": int(self.n_probs),

            # We don't want sampled/post-sampler probabilities.
            "post_sampling_probs": False,

            # Avoid repetition penalties affecting the reported distribution.
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,

            # Disable sampling filters as an additional precaution.
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,

            # Decider's local GGUF path clears memory between rows.
            # false gives behavior closer to that path.
            "cache_prompt": self.cache_prompt,

            "stream": False,
        }

        r = self.session.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()

        body = r.json()
        self.stats["requests"] += 1

        try:
            # Current llama.cpp response:
            #
            # "probs": [{
            #   "id": ...,
            #   "logprob": ...,
            #   "token": " A",
            #   "top_logprobs": [
            #       {"id": 32, "logprob": -0.12, ...},
            #       ...
            #   ]
            # }]
            #
            top = body["probs"][0]["top_logprobs"]

        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Unexpected llama.cpp /completion response; "
                f"expected probs[0].top_logprobs. Response: {body!r}"
            ) from exc

        # Match by token ID rather than token string.
        by_id = {
            int(entry["id"]): float(entry["logprob"])
            for entry in top
        }

        missing = [
            token_id
            for token_id in label_token_ids
            if int(token_id) not in by_id
        ]

        if missing:
            missing_info = []

            for token_id in missing:
                try:
                    piece = self.tok.decode([int(token_id)])
                except Exception:
                    piece = "?"

                missing_info.append(
                    f"{token_id}={piece!r}"
                )

            raise RuntimeError(
                "llama.cpp n_probs did not include all required Decider "
                "label tokens. Missing: "
                + ", ".join(missing_info)
                + f". Current n_probs={self.n_probs}. "
                  "Increase n_probs or use exact_n_probs=True."
            )

        # logprob = log(softmax(raw logits)).
        #
        # Differences between logprobs are exactly the same as differences
        # between logits:
        #
        #   log p_i = logit_i - logsumexp(all logits)
        #
        # Decider's later softmax only depends on relative logits, so these
        # values can safely stand in for raw logits.
        return np.asarray(
            [by_id[int(t)] for t in label_token_ids],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Decider Engine API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_items(
        self,
        items: list[dict[str, Any]],
        temperature=1.0,
    ):
        """
        Same contract as decider.engine_gguf.GGUFEngine.score_items().

        Returns:
            list[Tensor]

        Each tensor shape:
            [number_of_slots, MAX_OPTIONS]
        """

        results = []

        for item in items:
            slots = item["slots"]

            # /completion exposes only the distribution immediately after the
            # END of its prompt. It cannot expose arbitrary intermediate
            # answer-slot logits from one request.
            #
            # For System One independent=True, every item naturally has one
            # slot and that slot is the final prompt token.
            if len(slots) != 1:
                raise NotImplementedError(
                    "Remote llama.cpp /completion requires one Decider "
                    "answer slot per item. Use system_one(..., "
                    "independent=True)."
                )

            slot = int(slots[0])

            if slot != len(item["ids"]) - 1:
                raise RuntimeError(
                    "Decider answer slot is not at the end of the prompt; "
                    "/completion cannot read an intermediate-position logit."
                )

            nopts = int(item["nopts"][0])

            label_ids = [
                int(x)
                for x in self.letters[:nopts]
            ]

            logits = self._completion_logits(
                item["ids"],
                label_ids,
            )

            # Decider expects MAX_OPTIONS columns.
            full = np.full(
                MAX_OPTIONS,
                -np.inf,
                dtype=np.float32,
            )
            full[:nopts] = logits

            results.append(
                torch.from_numpy(full[None, :])
            )

        self.stats["forwards"] += len(items)

        # This is intentionally the same final stage used by GGUFEngine.
        #
        # It applies:
        #   Choice temperature
        #   Noul temperature
        #   Score temperature
        #
        # according to the item's types.
        all_logits = torch.cat(results, dim=0)

        nopts = torch.tensor(
            [
                n
                for item in items
                for n in item["nopts"]
            ]
        )

        all_logits = all_logits.masked_fill(
            torch.arange(MAX_OPTIONS)[None, :]
            >= nopts[:, None],
            float("-inf"),
        )

        probs = scaled_softmax(
            all_logits,
            slot_temperatures(
                temperature,
                items,
            ),
        )

        sizes = [
            len(item["slots"])
            for item in items
        ]

        return list(torch.split(probs, sizes))

    def score_shared(
        self,
        items,
        temperature=1.0,
        min_prefix=192,
    ):
        """
        Local Decider uses shared-prefix KV handling here.

        A remote llama.cpp server controls its own KV cache, so simply score
        each row independently.
        """
        return self.score_items(
            items,
            temperature=temperature,
        )

    def warmup(self, *args, **kwargs):
        return 0.0
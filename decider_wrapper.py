import json

from decider.infer import Decider
from decider import temperature as TT
from decider.prompt import (
    load_decider_config,
    resolve_layout,
    chat_template,
)

from remote_llamacpp_engine import RemoteLlamaCppEngine


class RemoteDecider(Decider):

    def __init__(
        self,
        model_dir,
        llama_url,
        *,
        n_probs=256,
        exact_n_probs=False,
    ):
        cfg = load_decider_config(model_dir)

        self.layout = resolve_layout(cfg)

        (self.T, self.T_by_type), (
            self.T_schema,
            self.T_schema_by_type,
        ) = TT.from_config(
            cfg,
            None,
            None,
        )

        self.neutralize_none = bool(
            cfg.get("neutralize_none", True)
        )

        self.eng = RemoteLlamaCppEngine(
            llama_url,
            model_dir,
            n_probs=n_probs,
            exact_n_probs=exact_n_probs,
        )

        self.m = self.eng.m

        self.chat = (
            chat_template(self.m.tok)
            if self.layout == "chat"
            else None
        )

        self.dev = "remote-llama.cpp"

        self.name = (
            "decider-"
            + str(cfg.get("version", "dev"))
        )

        self.schema_first = False

        self.isolated_levels = bool(
            cfg.get("isolated_levels", False)
        )

        self.abstain_below = 0.0

        self.gguf = True

        self._se = None
        self._schemas = {}
from __future__ import annotations

from typing import Any


def load_text_causal_lm(
    model_name: str,
    *,
    revision: str | None,
    local_files_only: bool,
    dtype: Any,
    device: str,
) -> Any:
    """Load only Qwen3.5's language component when the checkpoint is multimodal."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "local_files_only": local_files_only,
        "dtype": dtype,
    }
    if device != "cpu":
        load_kwargs["device_map"] = {"": device}

    if getattr(config, "model_type", None) == "qwen3_5" and hasattr(
        config, "text_config"
    ):
        from transformers import Qwen3_5ForCausalLM

        # Official Qwen3.5 multimodal checkpoints store the language component
        # below model.language_model. The text-only class avoids retaining the
        # unused vision tower during text SFT and inference.
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_name,
            config=config.text_config,
            key_mapping={r"^model\.language_model\.": "model."},
            **load_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if device == "cpu":
        model.to(device)
    return model

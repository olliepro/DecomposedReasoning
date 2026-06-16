#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)


@dataclass(frozen=True)
class ExportPaths:
    source_dir: Path
    output_dir: Path


def parse_args() -> ExportPaths:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return ExportPaths(source_dir=args.source_dir, output_dir=args.output_dir)


def make_qwen3_config(source_dir: Path) -> Qwen3Config:
    full_config = AutoConfig.from_pretrained(source_dir, trust_remote_code=True)
    text_values = full_config.text_config.to_dict()
    allowed_keys = set(Qwen3Config().to_dict())
    config_values = {key: value for key, value in text_values.items() if key in allowed_keys}

    for rope_key in ("rope_parameters", "rope_scaling"):
        rope_values = config_values.get(rope_key)
        if isinstance(rope_values, dict):
            rope_values.pop("mrope_section", None)
            rope_values.pop("mrope_interleaved", None)
            if rope_values == {"rope_type": "default"}:
                config_values.pop(rope_key)

    config_values["model_type"] = "qwen3"
    qwen3_config = Qwen3Config(**config_values)
    qwen3_config.architectures = ["Qwen3ForCausalLM"]
    return qwen3_config


def source_weight_files(source_dir: Path) -> list[Path]:
    single_file = source_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]
    files = sorted(source_dir.glob("model-*-of-*.safetensors"))
    assert files, f"no safetensors files found under {source_dir}"
    return files


def map_language_weights(source_dir: Path, qwen3_config: Qwen3Config) -> dict[str, torch.Tensor]:
    mapped_state: dict[str, torch.Tensor] = {}

    for source_file in source_weight_files(source_dir=source_dir):
        with safe_open(source_file, framework="pt", device="cpu") as source_state:
            for key in source_state.keys():
                if key.startswith("model.language_model."):
                    mapped_key = "model." + key.removeprefix("model.language_model.")
                    mapped_state[mapped_key] = source_state.get_tensor(key)
                elif key == "lm_head.weight":
                    mapped_state[key] = source_state.get_tensor(key)

    with torch.device("meta"):
        expected_keys = set(Qwen3ForCausalLM(qwen3_config).state_dict())
    actual_keys = set(mapped_state)
    assert actual_keys == expected_keys, (
        f"mapped keys do not match Qwen3ForCausalLM: "
        f"missing={sorted(expected_keys - actual_keys)[:20]} "
        f"extra={sorted(actual_keys - expected_keys)[:20]}"
    )
    return mapped_state


def save_text_model(paths: ExportPaths) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    qwen3_config = make_qwen3_config(source_dir=paths.source_dir)
    mapped_state = map_language_weights(source_dir=paths.source_dir, qwen3_config=qwen3_config)

    qwen3_config.save_pretrained(paths.output_dir)
    save_file(mapped_state, paths.output_dir / "model.safetensors", metadata={"format": "pt"})
    del mapped_state
    gc.collect()

    AutoTokenizer.from_pretrained(paths.source_dir, trust_remote_code=True).save_pretrained(paths.output_dir)
    GenerationConfig.from_pretrained(paths.source_dir).save_pretrained(paths.output_dir)

    loaded = AutoModelForCausalLM.from_pretrained(
        paths.output_dir,
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    assert loaded.config.model_type == "qwen3"
    assert loaded.config.architectures == ["Qwen3ForCausalLM"]


def main() -> None:
    save_text_model(paths=parse_args())


if __name__ == "__main__":
    main()

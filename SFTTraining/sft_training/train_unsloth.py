"""Train one SFT run with Unsloth LoRA in 16-bit weights."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer

from sft_training.config_types import RunConfig
from sft_training.train import (
    _enforce_supervision_rows,
    build_datasets,
    configure_process_cuda_device,
    compute_warmup_steps,
    init_wandb_run,
    is_world_process_zero,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Unsloth training entrypoint.

    Returns:
        Parsed argument namespace for config-driven training.
    """
    parser = argparse.ArgumentParser(description="Train one Unsloth SFT run.")
    parser.add_argument("--config", type=Path, required=True, help="Run YAML config path.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--override-num-epochs", type=int, default=None)
    parser.add_argument("--override-max-seq-length", type=int, default=None)
    return parser.parse_args()


def apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Apply CLI overrides without mutating the loaded config.

    Args:
        config: Parsed run configuration.
        args: CLI override arguments.

    Returns:
        Updated run configuration with CLI overrides applied.
    """
    override_epochs = args.override_num_epochs or config.num_train_epochs
    override_max_seq_length = args.override_max_seq_length or config.max_seq_length
    return replace(
        config,
        num_train_epochs=int(override_epochs),
        max_seq_length=int(override_max_seq_length),
    )


def load_unsloth_model(config: RunConfig) -> tuple[Any, Any]:
    """Load an Unsloth 16-bit model and attach LoRA adapters.

    Args:
        config: Run config containing model and LoRA settings.

    Returns:
        `(model, tokenizer)` ready for `SFTTrainer`.

    Example:
        >>> cfg = RunConfig.from_yaml(Path("configs/runs/qwen3_8b_to_think_merged_2213_unsloth.yaml"))
        >>> model, tokenizer = load_unsloth_model(config=cfg)
    """
    assert config.lora is not None, "Unsloth training requires a LoRA config."
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map: str | dict[str, int] = "sequential"
    if torch.cuda.is_available():
        device_map = {"": local_rank}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name_or_path,
        max_seq_length=config.max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        load_in_4bit=False,
        load_in_16bit=True,
        device_map=device_map,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=config.lora.r,
        target_modules=list(config.lora.target_modules),
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer = configure_chat_tokenizer(tokenizer=tokenizer)
    return model, tokenizer


def configure_chat_tokenizer(tokenizer: Any) -> Any:
    """Apply Unsloth's canonical Qwen 3 chat template to the tokenizer.

    Args:
        tokenizer: Loaded tokenizer for the target chat model.

    Returns:
        Tokenizer configured with template metadata used by
        `train_on_responses_only`.
    """
    return get_chat_template(
        tokenizer=tokenizer,
        chat_template="qwen3",
        mapping={
            "role": "role",
            "content": "content",
            "user": "user",
            "assistant": "assistant",
        },
        map_eos_token=True,
    )


def build_training_args(config: RunConfig, warmup_steps: int) -> SFTConfig:
    """Build trainer arguments for the Unsloth path.

    Args:
        config: Parsed run config.
        warmup_steps: Warmup step count computed from dataset size.

    Returns:
        `SFTConfig` configured for single-node Unsloth LoRA training.
    """
    return SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        optim="adamw_8bit",
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        seed=config.seed,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=config.save_total_limit,
        save_only_model=config.save_only_model,
        report_to=["wandb"],
        run_name=config.run_name,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=False,
        completion_only_loss=False,
    )


def build_conversation_dataset(dataset: Dataset) -> Dataset:
    """Convert prompt/completion rows into canonical conversation rows.

    Args:
        dataset: HuggingFace dataset with `prompt` and `completion` columns.

    Returns:
        Dataset with `conversations` plus row metadata.
    """
    conversation_rows: list[dict[str, object]] = []
    for row in dataset:
        conversation_rows.append(
            {
                "id": str(row["id"]),
                "dataset_source": str(row["dataset_source"]),
                "conversations": list(row["prompt"]) + list(row["completion"]),
            }
        )
    return Dataset.from_list(conversation_rows)


def render_chat_text_dataset(dataset: Dataset, tokenizer: Any) -> Dataset:
    """Render conversation rows into a `text` dataset using the chat template.

    Args:
        dataset: HuggingFace dataset with a `conversations` column.
        tokenizer: Tokenizer configured with Unsloth chat metadata.

    Returns:
        Dataset with `text` plus passthrough metadata columns.
    """

    def format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = [
            tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation in batch["conversations"]
        ]
        return {"text": texts}

    return dataset.map(format_batch, batched=True)


def build_trainer(config: RunConfig, train_dataset: Any, eval_dataset: Any) -> SFTTrainer:
    """Construct the Unsloth-backed `SFTTrainer`.

    Args:
        config: Parsed run config.
        train_dataset: Training dataset.
        eval_dataset: Eval dataset.

    Returns:
        Configured trainer instance.
    """
    model, tokenizer = load_unsloth_model(config=config)
    conversation_train_dataset = build_conversation_dataset(dataset=train_dataset)
    conversation_eval_dataset = build_conversation_dataset(dataset=eval_dataset)
    rendered_train_dataset = render_chat_text_dataset(
        dataset=conversation_train_dataset,
        tokenizer=tokenizer,
    )
    rendered_eval_dataset = render_chat_text_dataset(
        dataset=conversation_eval_dataset,
        tokenizer=tokenizer,
    )
    warmup_steps = compute_warmup_steps(
        config=config,
        train_dataset_size=len(rendered_train_dataset),
    )
    trainer = SFTTrainer(
        model=model,
        args=build_training_args(config=config, warmup_steps=warmup_steps),
        train_dataset=rendered_train_dataset,
        eval_dataset=rendered_eval_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
    )
    trainer = train_on_responses_only(
        trainer=trainer,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train_dataset = _enforce_supervision_rows(
        dataset=trainer.train_dataset,
        dataset_name="train",
    )
    trainer.eval_dataset = _enforce_supervision_rows(
        dataset=trainer.eval_dataset,
        dataset_name="eval",
    )
    return trainer


def run_training(config: RunConfig, max_train_samples: int | None) -> None:
    """Execute one configured Unsloth SFT run.

    Args:
        config: Parsed run config.
        max_train_samples: Optional row cap for smoke tests.

    Returns:
        None.
    """
    configure_process_cuda_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, eval_dataset = build_datasets(
        config=config,
        max_train_samples=max_train_samples,
    )
    init_wandb_run(config=config)
    trainer = build_trainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir=str(config.output_dir / "final_model"))
    if is_world_process_zero() and wandb.run is not None:
        wandb.finish()


def main() -> None:
    """CLI entrypoint for Unsloth SFT training."""
    args = parse_args()
    config = RunConfig.from_yaml(yaml_path=args.config)
    overridden_config = apply_overrides(config=config, args=args)
    run_training(
        config=overridden_config,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()

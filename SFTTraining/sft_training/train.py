"""Train SFT runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, cast

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset, IterableDataset
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    ProgressCallback,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from sft_training.config_types import LoraConfig as RunLoraConfig
from sft_training.config_types import RunConfig
from sft_training.wandb_utils import build_wandb_run_context


@dataclass(frozen=True)
class PromptCompletionSample:
    """Prompt/completion training example with conversational message payloads.

    Args:
        sample_id: Unique row identifier.
        dataset_source: Source dataset name.
        prompt: Context messages prior to the final assistant response.
        completion: Final assistant response message.
    """

    sample_id: str
    dataset_source: str
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]

    def to_record(self) -> dict[str, object]:
        """Convert sample to HuggingFace Dataset row format.

        Returns:
            Dictionary row compatible with TRL prompt-completion datasets.
        """
        return {
            "id": self.sample_id,
            "dataset_source": self.dataset_source,
            "prompt": self.prompt,
            "completion": self.completion,
        }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training entrypoint.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train one SFT run.")
    parser.add_argument(
        "--config", type=Path, required=True, help="Run YAML config path."
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--override-num-epochs", type=int, default=None)
    parser.add_argument("--override-max-seq-length", type=int, default=None)
    return parser.parse_args()


def is_world_process_zero() -> bool:
    """Return whether the current process is global rank zero.

    Returns:
        True when this process should perform single-rank side effects.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: Input JSONL path.

    Returns:
        Parsed row dictionaries.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert isinstance(row, dict), "Each JSONL row must be a dict."
            rows.append(row)
    return rows


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    """Normalize one raw message dictionary into strict role/content strings.

    Args:
        message: Raw message payload from source JSONL.

    Returns:
        Normalized message with `role` and `content`.
    """
    role = str(message["role"])
    content = str(message["content"])
    return {"role": role, "content": content}


def _last_assistant_index(messages: list[dict[str, Any]]) -> int:
    """Find the final assistant message index.

    Args:
        messages: Conversation message list.

    Returns:
        Index of the last assistant message in `messages`.
    """
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    raise AssertionError("Conversation row is missing an assistant message.")


def build_sample(row: dict[str, Any]) -> PromptCompletionSample:
    """Convert one transformed dataset row into prompt-completion format.

    Args:
        row: Raw transformed dataset row containing `messages`.

    Returns:
        Prompt-completion sample supervising only final assistant turn.

    Example:
        >>> row = {
        ...     "id": "x",
        ...     "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        ... }
        >>> build_sample(row=row).completion[0]["role"]
        'assistant'
    """
    messages = row.get("messages")
    assert isinstance(messages, list), "Expected `messages` to be a list."
    final_index = _last_assistant_index(messages=messages)
    prompt = [_normalize_message(message=m) for m in messages[:final_index]]
    completion = [_normalize_message(message=messages[final_index])]
    sample_id = str(row.get("id", "missing-id"))
    dataset_source = str(row.get("dataset_source", "unknown"))
    return PromptCompletionSample(
        sample_id=sample_id,
        dataset_source=dataset_source,
        prompt=prompt,
        completion=completion,
    )


def split_samples(
    samples: list[PromptCompletionSample],
    eval_split_ratio: float,
    seed: int,
) -> tuple[list[PromptCompletionSample], list[PromptCompletionSample]]:
    """Split samples into train/eval with a deterministic shuffled partition.

    Args:
        samples: Input prompt-completion samples.
        eval_split_ratio: Fraction of samples reserved for eval loss.
        seed: Deterministic shuffle seed.

    Returns:
        `(train_samples, eval_samples)` lists.
    """
    assert len(samples) > 1, "Need at least two samples for train/eval split."
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    eval_count = int(round(len(shuffled) * eval_split_ratio))
    eval_count = min(max(1, eval_count), len(shuffled) - 1)
    eval_samples = shuffled[:eval_count]
    train_samples = shuffled[eval_count:]
    return train_samples, eval_samples


def _effective_supervision_tokens(row: dict[str, Any]) -> int:
    """Count supervised labels that survive causal shift for one tokenized row.

    Args:
        row: Tokenized row containing `input_ids` and optional supervision masks.

    Returns:
        Number of valid labels after next-token shift.
    """
    input_ids = row.get("input_ids")
    assert isinstance(input_ids, list), "Tokenized row must contain list `input_ids`."
    if "completion_mask" in row:
        supervision_mask = [int(value) for value in row["completion_mask"]]
    elif "assistant_masks" in row:
        supervision_mask = [int(value) for value in row["assistant_masks"]]
    else:
        supervision_mask = [1] * len(input_ids)
    if "assistant_masks" in row and "completion_mask" in row:
        assistant_mask = [int(value) for value in row["assistant_masks"]]
        supervision_mask = [
            int(x and y) for x, y in zip(supervision_mask, assistant_mask)
        ]
    assert len(supervision_mask) == len(
        input_ids
    ), "Mask length must match `input_ids` length."
    return int(sum(supervision_mask[1:]))


def _filter_zero_supervision_rows(dataset: Dataset, dataset_name: str) -> Dataset:
    """Drop tokenized rows with zero supervised labels after causal shifting.

    Args:
        dataset: Tokenized HuggingFace dataset.
        dataset_name: Label used in progress logs.

    Returns:
        Dataset containing only rows with effective supervised labels.

    Example:
        >>> filtered = _filter_zero_supervision_rows(dataset=train_dataset, dataset_name="train")
        >>> len(filtered) <= len(train_dataset)
        True
    """
    keep_indices: list[int] = []
    dropped_sample_ids: list[str] = []
    for index, raw_row in enumerate(dataset):
        assert isinstance(raw_row, dict), "Tokenized dataset rows must be dict objects."
        row = cast(dict[str, Any], raw_row)
        if _effective_supervision_tokens(row=row) > 0:
            keep_indices.append(index)
            continue
        if len(dropped_sample_ids) < 5:
            dropped_sample_ids.append(str(row.get("id", f"{dataset_name}:{index}")))
    if len(keep_indices) == len(dataset):
        return dataset
    assert (
        keep_indices
    ), f"All rows were unsupervised in {dataset_name} after truncation."
    filtered_dataset = dataset.select(indices=keep_indices)
    return filtered_dataset


def _enforce_supervision_rows(
    dataset: Any,
    dataset_name: str,
) -> Any:
    """Filter rows that have no effective supervised labels.

    Args:
        dataset: Prepared trainer dataset.
        dataset_name: Name for logging.

    Returns:
        Filtered dataset of the same high-level type.
    """
    if dataset is None or not isinstance(dataset, Dataset):
        return dataset
    return _filter_zero_supervision_rows(dataset=dataset, dataset_name=dataset_name)


def build_datasets(
    config: RunConfig,
    max_train_samples: int | None,
) -> tuple[Dataset, Dataset]:
    """Load JSONL and build train/eval HuggingFace datasets.

    Args:
        config: Typed run configuration.
        max_train_samples: Optional row cap for smoke testing.

    Returns:
        Pair `(train_dataset, eval_dataset)` ready for `SFTTrainer`.
    """
    rows = read_jsonl(path=config.dataset_path)
    samples = [build_sample(row=row) for row in rows]
    if max_train_samples is not None:
        samples = samples[:max_train_samples]
    train_samples, eval_samples = split_samples(
        samples=samples,
        eval_split_ratio=config.eval_split_ratio,
        seed=config.seed,
    )
    train_dataset = Dataset.from_list([sample.to_record() for sample in train_samples])
    eval_dataset = Dataset.from_list([sample.to_record() for sample in eval_samples])
    return train_dataset, eval_dataset


def _wandb_config_payload(config: RunConfig) -> dict[str, Any]:
    """Build a JSON-serializable W&B config payload.

    Args:
        config: Typed run configuration.

    Returns:
        Plain dictionary with path values serialized as strings.
    """
    payload = asdict(config)
    payload["dataset_path"] = str(config.dataset_path)
    payload["output_dir"] = str(config.output_dir)
    payload["deepspeed_config_path"] = (
        None
        if config.deepspeed_config_path is None
        else str(config.deepspeed_config_path)
    )
    return payload


def init_wandb_run(config: RunConfig) -> None:
    """Initialize a single W&B run on global rank zero.

    Args:
        config: Typed run configuration.

    Returns:
        None.
    """
    if not is_world_process_zero():
        return
    run_context = build_wandb_run_context(
        base_run_name=config.run_name,
        job_type="train",
    )
    run_id = os.environ.get("SFT_WANDB_RUN_ID")
    init_kwargs: dict[str, Any] = {}
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_context.run_name,
        group=run_context.group_name,
        job_type=run_context.job_type,
        config=_wandb_config_payload(config=config),
        **init_kwargs,
    )


def load_tokenizer(config: RunConfig):
    """Load tokenizer and ensure it has a valid pad token.

    Args:
        config: Typed run configuration.

    Returns:
        Loaded tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        trust_remote_code=True,
    )
    return tokenizer


def load_model(
    config: RunConfig,
    tokenizer: Any,
) -> PreTrainedModel:
    """Load the causal LM and align tokenizer-driven special token settings.

    Args:
        config: Typed run configuration.
        tokenizer: Loaded tokenizer used for token-id alignment.

    Returns:
        Loaded model ready for process-local CUDA placement.
    """
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model_embeddings = cast(Any, model.get_input_embeddings())
    if len(tokenizer) != int(model_embeddings.num_embeddings):
        model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    for token_key in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, token_key)
        if token_id is not None:
            setattr(model.config, token_key, int(token_id))
            if hasattr(model, "generation_config"):
                setattr(model.generation_config, token_key, int(token_id))
    return model


def configure_process_cuda_device() -> None:
    """Select per-rank CUDA device when running distributed training.

    Returns:
        None.
    """
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)


def move_model_to_local_cuda(model: PreTrainedModel) -> PreTrainedModel:
    """Move the model to the local CUDA rank when GPUs are available.

    Args:
        model: Loaded model instance.

    Returns:
        Model moved to local CUDA device, or unchanged on CPU-only hosts.
    """
    if not torch.cuda.is_available():
        return model
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    module_model = cast(torch.nn.Module, model)
    module_model.to(device=torch.device(f"cuda:{local_rank}"))
    return model


def build_peft_lora_config(
    lora_config: RunLoraConfig | None,
) -> PeftLoraConfig | None:
    """Build optional PEFT LoRA config for trainer initialization.

    Args:
        lora_config: Parsed optional run-level LoRA settings.

    Returns:
        `PeftLoraConfig` when LoRA is enabled, else `None`.
    """
    if lora_config is None:
        return None
    valid_task_types = tuple(task_type.value for task_type in TaskType)
    assert lora_config.task_type in valid_task_types, "Invalid LoRA task_type."
    return PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=list(lora_config.target_modules),
        bias=lora_config.bias,
        task_type=TaskType(lora_config.task_type),
    )


def _world_size_from_env() -> int:
    """Read distributed world size from environment.

    Returns:
        Positive process world size integer.
    """
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def _deepspeed_kwargs(config: RunConfig) -> dict[str, Any]:
    """Build optional deepspeed kwargs for trainer args.

    Args:
        config: Typed run configuration.

    Returns:
        Mapping with deepspeed path when configured.
    """
    if config.deepspeed_config_path is None:
        return {}
    return {"deepspeed": str(config.deepspeed_config_path)}


def compute_warmup_steps(config: RunConfig, train_dataset_size: int) -> int:
    """Compute warmup steps from warmup ratio and effective optimizer steps.

    Args:
        config: Typed run configuration.
        train_dataset_size: Number of train rows.

    Returns:
        Warmup step count used by trainer args.
    """
    if config.warmup_ratio <= 0.0:
        return 0
    world_size = _world_size_from_env()
    global_batch = config.per_device_train_batch_size * world_size
    train_micro_steps = math.ceil(train_dataset_size / global_batch)
    optimizer_steps = math.ceil(train_micro_steps / config.gradient_accumulation_steps)
    total_steps = max(1, optimizer_steps * config.num_train_epochs)
    return max(1, int(total_steps * config.warmup_ratio))


def build_training_args(
    config: RunConfig,
    warmup_steps: int,
) -> SFTConfig:
    """Build TRL `SFTConfig` training arguments from run config.

    Args:
        config: Typed run configuration.
        warmup_steps: Number of scheduler warmup steps.

    Returns:
        Configured `SFTConfig` object.
    """
    return SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        optim=config.optim,
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
        logging_nan_inf_filter=False,
        save_total_limit=config.save_total_limit,
        save_only_model=config.save_only_model,
        report_to=["wandb"],
        run_name=config.run_name,
        bf16=True,
        gradient_checkpointing=True,
        completion_only_loss=True,
        **_deepspeed_kwargs(config=config),
    )


def build_trainer(
    config: RunConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    """Construct `SFTTrainer` for SFT loss/eval.

    Args:
        config: Typed run configuration.
        train_dataset: Training dataset.
        eval_dataset: Eval-loss dataset.

    Returns:
        Configured trainer instance.
    """
    tokenizer = load_tokenizer(config=config)
    loaded_model = load_model(
        config=config,
        tokenizer=tokenizer,
    )
    model = move_model_to_local_cuda(model=loaded_model)
    warmup_steps = compute_warmup_steps(
        config=config,
        train_dataset_size=len(train_dataset),
    )
    training_args = build_training_args(
        config=config,
        warmup_steps=warmup_steps,
    )
    peft_lora_config = build_peft_lora_config(lora_config=config.lora)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_lora_config,
    )
    trainer_any = cast(Any, trainer)
    trainer_any.train_dataset = _enforce_supervision_rows(
        dataset=trainer.train_dataset,
        dataset_name="train",
    )
    eval_dataset_payload = trainer.eval_dataset
    if isinstance(eval_dataset_payload, dict):
        trainer_any.eval_dataset = {
            dataset_key: _enforce_supervision_rows(
                dataset=dataset_value,
                dataset_name=f"eval/{dataset_key}",
            )
            for dataset_key, dataset_value in eval_dataset_payload.items()
        }
    else:
        trainer_any.eval_dataset = _enforce_supervision_rows(
            dataset=eval_dataset_payload,
            dataset_name="eval",
        )
    return trainer


def run_training(config: RunConfig, max_train_samples: int | None) -> None:
    """Execute one configured SFT training run.

    Args:
        config: Typed run configuration.
        max_train_samples: Optional smoke-test cap.

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


def apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Apply CLI overrides without mutating loaded config object.

    Args:
        config: Parsed run configuration.
        args: CLI override arguments.

    Returns:
        Updated run configuration object.
    """
    override_epochs = args.override_num_epochs or config.num_train_epochs
    override_max_seq_length = args.override_max_seq_length or config.max_seq_length
    return replace(
        config,
        num_train_epochs=int(override_epochs),
        max_seq_length=int(override_max_seq_length),
    )


def main() -> None:
    """CLI entrypoint for single-run SFT training."""
    args = parse_args()
    config = RunConfig.from_yaml(yaml_path=args.config)
    overridden_config = apply_overrides(config=config, args=args)
    run_training(
        config=overridden_config,
        max_train_samples=args.max_train_samples,
    )


if __name__ == "__main__":
    main()

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
    GenerationConfig,
    PreTrainedModel,
    ProgressCallback,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from sft_training.config_types import LoraConfig as RunLoraConfig
from sft_training.config_types import RunConfig
from sft_training.non_sequitur_masking import MaskTargets
from sft_training.non_sequitur_masking import build_assistant_tokenized_record
from sft_training.non_sequitur_masking import extract_mask_targets
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


@dataclass(frozen=True)
class ConversationSample:
    """Full-conversation training example retaining all chat messages.

    Args:
        sample_id: Unique row identifier.
        dataset_source: Source dataset name.
        messages: Entire normalized conversation message list.
        mask_targets: Optional non-sequitur masking metadata.
    """

    sample_id: str
    dataset_source: str
    messages: list[dict[str, str]]
    mask_targets: MaskTargets | None = None

    def to_record(self) -> dict[str, object]:
        """Convert sample to a conversational HuggingFace Dataset row.

        Returns:
            Dictionary row compatible with TRL conversational datasets.
        """
        record: dict[str, object] = {
            "id": self.sample_id,
            "dataset_source": self.dataset_source,
            "messages": self.messages,
        }
        if self.mask_targets is not None:
            record["mask_targets"] = self.mask_targets.to_record()
        return record


TrainingSample = PromptCompletionSample | ConversationSample


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


def _normalized_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize all conversation messages in one raw dataset row.

    Args:
        row: Raw transformed dataset row containing `messages`.

    Returns:
        Normalized conversation message list.
    """
    messages = row.get("messages")
    assert isinstance(messages, list), "Expected `messages` to be a list."
    return [
        _normalize_message(message=cast(dict[str, Any], message))
        for message in messages
    ]


def _sample_metadata(row: dict[str, Any]) -> tuple[str, str]:
    """Extract canonical sample metadata from a raw dataset row.

    Args:
        row: Raw transformed dataset row.

    Returns:
        `(sample_id, dataset_source)` tuple for downstream dataset records.
    """
    sample_id = str(row.get("id", "missing-id"))
    dataset_source = str(row.get("dataset_source", "unknown"))
    return sample_id, dataset_source


def sanitize_generation_config(generation_config: GenerationConfig) -> None:
    """Reset invalid greedy-generation flags to validator-safe defaults.

    Args:
        generation_config: Pretrained generation config attached to the model.

    Returns:
        None. The config is updated in place.

    Example:
        >>> config = GenerationConfig(do_sample=False, temperature=0.6, top_p=0.95)
        >>> sanitize_generation_config(generation_config=config)
        >>> (config.temperature, config.top_p)
        (1.0, 1.0)
    """
    if generation_config.do_sample is False:
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.min_p = None
        generation_config.typical_p = 1.0
        generation_config.top_k = 50
        generation_config.epsilon_cutoff = 0.0
        generation_config.eta_cutoff = 0.0
    if generation_config.num_beams == 1:
        generation_config.early_stopping = False
        generation_config.length_penalty = 1.0


def build_completion_only_sample(row: dict[str, Any]) -> PromptCompletionSample:
    """Convert one row into a prompt-completion training example.

    Args:
        row: Raw transformed dataset row containing `messages`.

    Returns:
        Prompt-completion sample supervising only final assistant turn.

    Example:
        >>> row = {
        ...     "id": "x",
        ...     "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        ... }
        >>> build_completion_only_sample(row=row).completion[0]["role"]
        'assistant'
    """
    messages = _normalized_messages(row=row)
    final_index = _last_assistant_index(messages=messages)
    prompt = messages[:final_index]
    completion = [messages[final_index]]
    sample_id, dataset_source = _sample_metadata(row=row)
    return PromptCompletionSample(
        sample_id=sample_id,
        dataset_source=dataset_source,
        prompt=prompt,
        completion=completion,
    )


def build_full_conversation_sample(
    row: dict[str, Any],
    include_mask_targets: bool = False,
) -> ConversationSample:
    """Convert one row into a full-conversation training example.

    Args:
        row: Raw transformed dataset row containing `messages`.
        include_mask_targets: Whether to preserve non-sequitur mask metadata.

    Returns:
        Full-conversation sample that keeps the complete chat intact.
    """
    sample_id, dataset_source = _sample_metadata(row=row)
    mask_targets = extract_mask_targets(row=row) if include_mask_targets else None
    return ConversationSample(
        sample_id=sample_id,
        dataset_source=dataset_source,
        messages=_normalized_messages(row=row),
        mask_targets=mask_targets,
    )


def build_sample(
    row: dict[str, Any],
    supervision_mode: str,
    include_mask_targets: bool = False,
) -> TrainingSample:
    """Build one training sample using the configured supervision mode.

    Args:
        row: Raw transformed dataset row.
        supervision_mode: Dataset supervision style for SFT.
        include_mask_targets: Whether to preserve non-sequitur mask metadata.

    Returns:
        Prompt-completion or full-conversation sample object.
    """
    if supervision_mode in ("full_conversation", "assistant_only"):
        return build_full_conversation_sample(
            row=row,
            include_mask_targets=include_mask_targets,
        )
    return build_completion_only_sample(row=row)


def uses_conversation_dataset(supervision_mode: str) -> bool:
    """Return whether a supervision mode should emit conversational rows.

    Args:
        supervision_mode: Dataset supervision style for SFT.

    Returns:
        True when training should use `messages` rows.
    """
    return supervision_mode in ("full_conversation", "assistant_only")


def split_samples(
    samples: list[TrainingSample],
    eval_split_ratio: float,
    seed: int,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
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
    samples = [
        build_sample(
            row=row,
            supervision_mode=config.supervision_mode,
            include_mask_targets=config.mask_non_sequitur_steer_spans,
        )
        for row in rows
    ]
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
    payload["chat_template_path"] = (
        None if config.chat_template_path is None else str(config.chat_template_path)
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
    """Load tokenizer and apply any configured training chat template.

    Args:
        config: Typed run configuration.

    Returns:
        Loaded tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name_or_path,
        trust_remote_code=True,
    )
    if config.chat_template_path is not None:
        tokenizer.chat_template = config.chat_template_path.read_text(encoding="utf-8")
    return tokenizer


def _tokenize_masked_conversation_example(
    example: dict[str, Any],
    tokenizer: Any,
) -> dict[str, list[int]]:
    """Tokenize one conversation row with optional non-sequitur masking.

    Args:
        example: Conversational dataset row with `messages` and optional `mask_targets`.
        tokenizer: Active training tokenizer.

    Returns:
        Tokenized row payload containing `input_ids` and `assistant_masks`.
    """
    messages = cast(list[dict[str, str]], example["messages"])
    return build_assistant_tokenized_record(
        tokenizer=tokenizer,
        messages=messages,
        mask_targets=extract_mask_targets(row=example),
    )


def _pretokenize_non_sequitur_dataset(
    dataset: Dataset,
    tokenizer: Any,
    dataset_name: str,
) -> Dataset:
    """Pretokenize one conversational dataset with non-sequitur masking.

    Args:
        dataset: HuggingFace conversational dataset.
        tokenizer: Active training tokenizer.
        dataset_name: Human-readable dataset label for progress logs.

    Returns:
        Pretokenized dataset with `assistant_masks` ready for TRL packing.
    """
    remove_columns = [
        column_name
        for column_name in ("messages", "mask_targets")
        if column_name in dataset.column_names
    ]
    return dataset.map(
        _tokenize_masked_conversation_example,
        fn_kwargs={"tokenizer": tokenizer},
        batched=False,
        remove_columns=remove_columns,
        desc=f"Tokenizing {dataset_name} dataset with non-sequitur masking",
    )


def _maybe_pretokenize_non_sequitur_datasets(
    config: RunConfig,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> tuple[Dataset, Dataset]:
    """Pretokenize train/eval datasets when non-sequitur masking is enabled.

    Args:
        config: Typed run configuration.
        tokenizer: Active training tokenizer.
        train_dataset: Train split before TRL preprocessing.
        eval_dataset: Eval split before TRL preprocessing.

    Returns:
        Train/eval datasets, pretokenized only when masking is enabled.
    """
    if not config.mask_non_sequitur_steer_spans:
        return train_dataset, eval_dataset
    assert (
        config.supervision_mode == "assistant_only"
    ), "Non-sequitur masking only supports assistant-only supervision."
    return (
        _pretokenize_non_sequitur_dataset(
            dataset=train_dataset,
            tokenizer=tokenizer,
            dataset_name="train",
        ),
        _pretokenize_non_sequitur_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_name="eval",
        ),
    )


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
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": config.model_name_or_path,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(
        **model_kwargs,
    )
    if model.generation_config is not None:
        sanitize_generation_config(generation_config=model.generation_config)
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


def move_model_to_local_cuda(
    model: PreTrainedModel, config: RunConfig
) -> PreTrainedModel:
    """Move the model to the local CUDA rank when DeepSpeed is not active.

    Args:
        model: Loaded model instance.
        config: Parsed run configuration.

    Returns:
        Model moved to local CUDA device, or unchanged when CPU-only or
        when DeepSpeed handles placement.

    Example:
        >>> moved_model = move_model_to_local_cuda(model=model, config=config)
        >>> moved_model is model
        True
    """
    if not torch.cuda.is_available() or config.deepspeed_config_path is not None:
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
        chat_template_path=(
            None
            if config.chat_template_path is None
            else str(config.chat_template_path)
        ),
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
        gradient_checkpointing=config.gradient_checkpointing,
        activation_offloading=config.activation_offloading,
        packing=config.packing,
        packing_strategy=config.packing_strategy,
        padding_free=config.padding_free,
        eval_packing=config.eval_packing,
        completion_only_loss=(config.supervision_mode == "completion_only"),
        assistant_only_loss=uses_trl_assistant_only_loss(config=config),
        **_deepspeed_kwargs(config=config),
    )


def uses_trl_assistant_only_loss(config: RunConfig) -> bool:
    """Return whether TRL should derive assistant masks internally.

    Args:
        config: Typed run configuration.

    Returns:
        True only for unmasked assistant-only runs. Masked runs pretokenize
        their own `assistant_masks`, so TRL should not re-enter its
        conversational assistant-only path.
    """
    return (
        config.supervision_mode == "assistant_only"
        and not config.mask_non_sequitur_steer_spans
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
    train_dataset, eval_dataset = _maybe_pretokenize_non_sequitur_datasets(
        config=config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    loaded_model = load_model(
        config=config,
        tokenizer=tokenizer,
    )
    model = move_model_to_local_cuda(model=loaded_model, config=config)
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

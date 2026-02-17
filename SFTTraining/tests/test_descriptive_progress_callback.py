"""Unit tests for descriptive training progress callback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from transformers import TrainerControl, TrainerState, TrainingArguments

from sft_training.callbacks.descriptive_progress_callback import (
    DescriptiveProgressCallback,
    progress_description,
    row_progress,
)


@dataclass
class FakeProgressBar:
    """Simple tqdm stand-in for callback behavior tests."""

    updated_steps: int = 0
    description: str = ""
    postfix: dict[str, str] | None = None
    refresh: bool | None = None

    def update(self, steps: int) -> None:
        """Record incremental step updates."""
        self.updated_steps += steps

    def set_description(self, description: str) -> None:
        """Record latest bar description."""
        self.description = description

    def set_postfix(self, postfix: dict[str, str], refresh: bool = True) -> None:
        """Record latest bar postfix payload."""
        self.postfix = postfix
        self.refresh = refresh


def test_progress_description_epoch_boundary_behavior() -> None:
    """Description should retain epoch 1 at `state.epoch=1.0`."""
    state = TrainerState(global_step=10, max_steps=100, epoch=1.0)
    description = progress_description(state=state, total_epochs=8)
    assert description == "epoch 1/8 | step 10/100"


def test_row_progress_caps_processed_rows(tmp_path: Path) -> None:
    """Row progress should cap processed rows at total rows."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
    )
    state = TrainerState(global_step=999, max_steps=1000)
    rows = row_progress(state=state, args=args, total_train_rows=300)
    assert rows.processed_rows == 300
    assert rows.total_rows == 300


def test_callback_updates_description_and_postfix(tmp_path: Path) -> None:
    """Callback should set readable epoch/step description and rows postfix."""
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=3,
    )
    state = TrainerState(global_step=5, max_steps=50, epoch=1.2, is_world_process_zero=True)
    control = TrainerControl()
    callback = DescriptiveProgressCallback(train_dataset_size=100, total_epochs=8)
    callback_any = cast(Any, callback)
    callback_any.training_bar = FakeProgressBar()
    callback_any.current_step = 0

    callback.on_step_end(args=args, state=state, control=control)

    progress_bar = cast(FakeProgressBar, callback_any.training_bar)
    assert progress_bar.description == "epoch 2/8 | step 5/50"
    assert progress_bar.postfix == {"rows": "30/800"}
    assert progress_bar.updated_steps == 5

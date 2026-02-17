"""Descriptive tqdm progress callback for training loops."""

from __future__ import annotations

import math
from dataclasses import dataclass

from transformers import ProgressCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass(frozen=True)
class RowProgress:
    """Display-friendly training row progress values.

    Args:
        processed_rows: Number of training rows consumed so far.
        total_rows: Total training rows across all epochs.
    """

    processed_rows: int
    total_rows: int

    def to_postfix(self) -> dict[str, str]:
        """Return tqdm postfix values.

        Returns:
            Mapping used by `tqdm.set_postfix`.
        """
        return {"rows": f"{self.processed_rows}/{self.total_rows}"}


def epoch_index(state: TrainerState, total_epochs: int) -> int:
    """Compute a 1-based epoch index for progress display.

    Args:
        state: Trainer callback state.
        total_epochs: Configured total epoch count.

    Returns:
        Current epoch index in the range `[1, total_epochs]`.
    """
    epoch_value = 0.0 if state.epoch is None else float(state.epoch)
    return min(total_epochs, max(1, int(math.ceil(epoch_value))))


def progress_description(state: TrainerState, total_epochs: int) -> str:
    """Build a concise tqdm description.

    Args:
        state: Trainer callback state.
        total_epochs: Configured total epoch count.

    Returns:
        Description string with epoch and step counters.

    Example:
        >>> from transformers import TrainerState
        >>> progress_description(state=TrainerState(global_step=8, max_steps=100, epoch=1.2), total_epochs=8)
        'epoch 2/8 | step 8/100'
    """
    max_steps = max(1, int(state.max_steps))
    current_epoch = epoch_index(state=state, total_epochs=total_epochs)
    return f"epoch {current_epoch}/{total_epochs} | step {state.global_step}/{max_steps}"


def row_progress(
    state: TrainerState,
    args: TrainingArguments,
    total_train_rows: int,
) -> RowProgress:
    """Compute approximate row progress from optimizer steps.

    Args:
        state: Trainer callback state.
        args: Training arguments containing batch and accumulation settings.
        total_train_rows: Total rows across all epochs.

    Returns:
        Row progress dataclass used for tqdm postfix display.
    """
    rows_per_step = (
        int(args.per_device_train_batch_size)
        * max(1, int(args.world_size))
        * int(args.gradient_accumulation_steps)
    )
    processed_rows = min(total_train_rows, state.global_step * max(1, rows_per_step))
    return RowProgress(processed_rows=processed_rows, total_rows=total_train_rows)


class DescriptiveProgressCallback(ProgressCallback):
    """Progress callback showing epoch/step and rows processed.

    Args:
        train_dataset_size: Number of training rows in one epoch.
        total_epochs: Number of training epochs.

    Example:
        >>> callback = DescriptiveProgressCallback(train_dataset_size=5000, total_epochs=8)
        >>> callback.total_train_rows
        40000
    """

    def __init__(self, train_dataset_size: int, total_epochs: int) -> None:
        super().__init__()
        self.train_dataset_size = max(1, int(train_dataset_size))
        self.total_epochs = max(1, int(total_epochs))
        self.total_train_rows = self.train_dataset_size * self.total_epochs

    def _set_progress_text(self, args: TrainingArguments, state: TrainerState) -> None:
        """Update tqdm description and postfix for the active training bar.

        Args:
            args: Training arguments for batch context.
            state: Trainer callback state.

        Returns:
            None.
        """
        if not state.is_world_process_zero or self.training_bar is None:
            return
        description = progress_description(state=state, total_epochs=self.total_epochs)
        rows = row_progress(state=state, args=args, total_train_rows=self.total_train_rows)
        self.training_bar.set_description(description)
        self.training_bar.set_postfix(rows.to_postfix(), refresh=False)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Initialize tqdm and set the initial descriptive text."""
        _ = kwargs
        super().on_train_begin(args=args, state=state, control=control)
        self._set_progress_text(args=args, state=state)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Advance tqdm and refresh description/postfix after each step."""
        _ = kwargs
        super().on_step_end(args=args, state=state, control=control)
        self._set_progress_text(args=args, state=state)

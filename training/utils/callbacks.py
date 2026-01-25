"""
Training callbacks for monitoring and logging.
"""

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class LoggingCallback(TrainerCallback):
    """Callback for enhanced logging during training."""

    def __init__(self, log_every_n_steps: int = 50):
        """
        Initialize logging callback.

        Args:
            log_every_n_steps: Log progress every N steps
        """
        self.log_every_n_steps = log_every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Log training progress."""
        if logs is None:
            return

        step = state.global_step
        if step % self.log_every_n_steps == 0:
            loss = logs.get("loss", logs.get("train_loss", "N/A"))
            lr = logs.get("learning_rate", "N/A")
            if isinstance(loss, float):
                print(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")
            else:
                print(f"Step {step}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log at end of epoch."""
        epoch = state.epoch
        print(f"\n{'='*50}")
        print(f"Epoch {epoch:.0f} completed")
        print(f"{'='*50}\n")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log at end of training."""
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Total steps: {state.global_step}")
        print(f"Best metric: {state.best_metric}")
        print("=" * 50 + "\n")


class MemoryCallback(TrainerCallback):
    """Callback for monitoring GPU memory usage."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log memory usage periodically."""
        if state.global_step % self.log_every_n_steps != 0:
            return

        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        except Exception:
            pass

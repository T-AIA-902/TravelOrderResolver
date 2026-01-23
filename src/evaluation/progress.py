"""
Progress tracking utilities for evaluation.

Provides a callback protocol and default implementations for
displaying evaluation progress.
"""

from typing import Callable, Protocol


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, current: int, total: int) -> None:
        """
        Called to report progress.

        Args:
            current: Current sample count (processed so far)
            total: Total number of samples
        """
        ...


def print_progress(current: int, total: int) -> None:
    """
    Default progress printer.

    Prints progress in format: "    Progress: X/Y (Z%)"

    Args:
        current: Current sample count
        total: Total number of samples
    """
    if total == 0:
        return
    pct = current * 100 / total
    print(f"    Progress: {current}/{total} ({pct:.0f}%)")


def create_progress_callback(
    batch_size: int = 128,
) -> Callable[[int, int], None]:
    """
    Create a progress callback that prints after each batch.

    Args:
        batch_size: Print progress every batch_size samples

    Returns:
        A callback function that tracks and prints progress
    """
    last_printed = [0]  # Use list to allow mutation in closure

    def callback(current: int, total: int) -> None:
        # Print at batch boundaries or at completion
        if current - last_printed[0] >= batch_size or current == total:
            print_progress(current, total)
            last_printed[0] = current

    return callback

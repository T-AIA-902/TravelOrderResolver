"""
Device detection utilities for GPU/CPU selection.

Provides automatic device detection for PyTorch and SpaCy models.
"""

from typing import Literal

DeviceType = Literal["auto", "cuda", "cpu"]


def get_torch_device(preferred: DeviceType = "auto") -> str:
    """
    Get the best available PyTorch device.

    Args:
        preferred: Device preference - "auto", "cuda", or "cpu"
            - "auto": Use CUDA if available, else CPU
            - "cuda": Force CUDA (will fail if not available)
            - "cpu": Force CPU

    Returns:
        Device string: "cuda" or "cpu"

    Raises:
        RuntimeError: If "cuda" is requested but not available
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not available, using CPU")
        return "cpu"

    if preferred == "cpu":
        return "cpu"

    cuda_available = torch.cuda.is_available()

    if preferred == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA requested but not available")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        return "cuda"

    # auto mode
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available, using GPU: {device_name}")
        return "cuda"
    else:
        print("CUDA not available, using CPU")
        return "cpu"


def setup_spacy_device(preferred: DeviceType = "auto") -> bool:
    """
    Configure SpaCy to use GPU if available.

    Must be called BEFORE loading any SpaCy model.

    Args:
        preferred: Device preference - "auto", "cuda", or "cpu"

    Returns:
        True if GPU is activated, False otherwise
    """
    if preferred == "cpu":
        print("SpaCy: Using CPU (as requested)")
        return False

    try:
        import spacy
    except ImportError:
        print("SpaCy not available")
        return False

    try:
        # prefer_gpu() returns True if GPU is available and activated
        if spacy.prefer_gpu():
            print("SpaCy: GPU activated")
            return True
        else:
            if preferred == "cuda":
                print("SpaCy: GPU requested but not available, using CPU")
            else:
                print("SpaCy: GPU not available, using CPU")
            return False
    except Exception as e:
        print(f"SpaCy: GPU setup failed ({e}), using CPU")
        return False


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device information
    """
    info: dict = {
        "torch_available": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "spacy_gpu_available": False,
    }

    try:
        import torch

        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import spacy

        # Check if spacy can use GPU without actually activating it
        info["spacy_gpu_available"] = spacy.prefer_gpu()
    except (ImportError, Exception):
        pass

    return info

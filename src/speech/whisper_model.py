"""
Whisper model wrapper for speech-to-text transcription.

Provides a high-level interface around OpenAI's Whisper model
for transcribing audio files to text, optimized for French.
"""

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
from loguru import logger

from src.utils.device import DeviceType, get_torch_device

WhisperModelSize = Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


@dataclass
class TranscriptionResult:
    """Result of a Whisper transcription."""

    text: str
    language: str
    segments: list[dict] = field(default_factory=list)


class WhisperModel:
    """
    Wrapper around OpenAI's Whisper model.

    Handles model loading, device selection, and transcription.
    The model is loaded lazily on first transcription or via load().

    Args:
        model_name: Whisper model size. Default "medium" for good French accuracy.
        device: Device preference for inference.
        language: Language hint for transcription. Default "fr" (French).
    """

    def __init__(
        self,
        model_name: WhisperModelSize = "medium",
        device: DeviceType = "auto",
        language: str = "fr",
    ):
        self._model_name = model_name
        self._device = get_torch_device(device)
        self._language = language
        self._model = None

    @property
    def is_loaded(self) -> bool:
        """Whether the Whisper model is currently loaded in memory."""
        return self._model is not None

    def load(self) -> None:
        """Load the Whisper model into memory."""
        if self._model is not None:
            logger.debug("Whisper model already loaded, skipping")
            return

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper is required for speech-to-text. "
                "Install with: poetry install --with ml"
            )

        logger.info(f"Loading Whisper model '{self._model_name}' on {self._device}...")
        self._model = whisper.load_model(self._model_name, device=self._device)
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio: Union[str, np.ndarray]) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Path to an audio file, or a numpy float32 array of audio samples.
                   When passing a file path, ffmpeg must be installed.
                   Passing a numpy array avoids the ffmpeg dependency.

        Returns:
            TranscriptionResult with text, detected language, and segments.
        """
        if self._model is None:
            self.load()

        logger.info("Transcribing audio...")
        result = self._model.transcribe(
            audio,
            language=self._language,
            fp16=(self._device == "cuda"),
        )

        transcription = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", self._language),
            segments=[
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
        )

        logger.info(f"Transcription complete: '{transcription.text[:80]}...'")
        return transcription

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded")

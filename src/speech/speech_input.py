"""
Speech input helper for the CLI.

Handles Whisper transcriber initialization and microphone recording
for the interactive command-line interface.
"""

import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.speech.whisper_model import WhisperModelSize


class SpeechInput:
    """Manages speech-to-text input for the CLI."""

    def __init__(self, model_name: "WhisperModelSize" = "medium"):
        self._transcriber = None
        self._available = False

        try:
            from src.speech.transcriber import SpeechTranscriber

            self._transcriber = SpeechTranscriber(model_name=model_name)
            self._transcriber.warmup()
            self._available = True
            print("Speech-to-text ready (Whisper medium)", file=sys.stderr)
        except ImportError:
            print(
                "Error: speech dependencies not installed. "
                "Run: pip install openai-whisper sounddevice soundfile",
                file=sys.stderr,
            )

    @property
    def available(self) -> bool:
        """Whether speech input is ready to use."""
        return self._available

    def record(self, duration: float = 5.0) -> Optional[str]:
        """
        Record from microphone and return transcribed text.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Transcribed text, or None if recording failed.
        """
        if not self._available or not self._transcriber:
            return None

        try:
            print("Parle maintenant (5 secondes)...")
            result = self._transcriber.transcribe_from_mic(duration=duration)
            text = result.text.strip()
            print(f"Transcription: {text}")
            if not text:
                print("(rien detecte, reessayez)", file=sys.stderr)
                return None
            return text
        except Exception as e:
            print(f"Erreur micro: {e}", file=sys.stderr)
            return None

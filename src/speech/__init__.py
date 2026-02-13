"""Speech-to-text module using OpenAI Whisper."""

from src.speech.audio_processor import AudioRecorder
from src.speech.transcriber import SpeechTranscriber
from src.speech.whisper_model import TranscriptionResult, WhisperModel

__all__ = [
    "AudioRecorder",
    "SpeechTranscriber",
    "TranscriptionResult",
    "WhisperModel",
]

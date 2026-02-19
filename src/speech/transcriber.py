"""
Speech-to-text transcriber orchestrating audio recording and Whisper inference.

Combines AudioRecorder and WhisperModel into a simple high-level interface
for microphone-based transcription.
"""

from src.speech.audio_processor import AudioRecorder
from src.speech.whisper_model import TranscriptionResult, WhisperModel, WhisperModelSize
from src.utils.device import DeviceType


class SpeechTranscriber:
    """
    High-level speech-to-text transcriber.

    Orchestrates microphone recording and Whisper transcription
    in a single convenient interface.

    Args:
        model_name: Whisper model size.
        device: Device preference for Whisper inference.
        language: Language hint for transcription.
        sample_rate: Audio sample rate in Hz.
    """

    def __init__(
        self,
        model_name: WhisperModelSize = "small",
        device: DeviceType = "auto",
        language: str = "fr",
        sample_rate: int = 16000,
    ):
        self._recorder = AudioRecorder(sample_rate=sample_rate)
        self._model = WhisperModel(
            model_name=model_name,
            device=device,
            language=language,
        )

    def warmup(self) -> None:
        """Pre-load the Whisper model to avoid latency on first transcription."""
        self._model.load()

    def transcribe_from_mic(self, duration: float = 5.0) -> TranscriptionResult:
        """
        Record audio from the microphone for a fixed duration and transcribe it.

        Args:
            duration: Recording duration in seconds.

        Returns:
            TranscriptionResult with the transcribed text.
        """
        audio = self._recorder.record(duration)
        return self._model.transcribe(audio)

    def transcribe_from_mic_auto(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        max_duration: float = 30.0,
    ) -> TranscriptionResult:
        """
        Record audio until silence is detected, then transcribe.

        Stops recording when audio level stays below the threshold
        for `silence_duration` seconds, or when `max_duration` is reached.

        Args:
            silence_threshold: RMS amplitude below which audio is considered silence.
            silence_duration: Seconds of continuous silence before stopping.
            max_duration: Maximum recording duration in seconds.

        Returns:
            TranscriptionResult with the transcribed text.
        """
        audio = self._recorder.record_until_silence(
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
        )
        return self._model.transcribe(audio)

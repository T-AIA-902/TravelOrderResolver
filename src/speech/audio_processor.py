"""
Audio recording from microphone.

Provides microphone capture with fixed-duration and silence-detection modes,
outputting audio in the format expected by Whisper (16kHz mono WAV).
"""

import tempfile

import numpy as np
from loguru import logger


class AudioRecorder:
    """
    Records audio from the system microphone.

    Outputs 16kHz mono float32 audio, which is the native format for Whisper.

    Args:
        sample_rate: Audio sample rate in Hz. Default 16000 (Whisper native).
        channels: Number of audio channels. Default 1 (mono).
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self._sample_rate = sample_rate
        self._channels = channels
        self._ensure_dependencies()

    @staticmethod
    def _ensure_dependencies() -> None:
        """Check that audio dependencies are available."""
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            raise ImportError(
                "sounddevice is required for microphone recording. "
                "Install with: poetry install --with ml"
            )
        try:
            import soundfile  # noqa: F401
        except ImportError:
            raise ImportError(
                "soundfile is required for audio file I/O. "
                "Install with: poetry install --with ml"
            )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def record(self, duration: float) -> np.ndarray:
        """
        Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Numpy array of shape (n_samples,) with float32 audio data.
        """
        import sounddevice as sd

        num_samples = int(duration * self._sample_rate)
        logger.info(f"Recording {duration}s of audio at {self._sample_rate}Hz...")

        audio = sd.rec(
            num_samples,
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
        )
        sd.wait()

        # Flatten to 1D if mono
        audio = audio.squeeze()
        logger.info(f"Recording complete: {len(audio)} samples")
        return audio

    def record_until_silence(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        max_duration: float = 30.0,
        chunk_duration: float = 0.5,
    ) -> np.ndarray:
        """
        Record audio until silence is detected or max duration is reached.

        Stops recording when the audio level stays below the threshold
        for at least `silence_duration` seconds.

        Args:
            silence_threshold: RMS amplitude below which audio is considered silence.
            silence_duration: Seconds of continuous silence before stopping.
            max_duration: Maximum recording duration in seconds.
            chunk_duration: Duration of each recording chunk in seconds.

        Returns:
            Numpy array of shape (n_samples,) with float32 audio data.
        """
        import sounddevice as sd

        chunk_samples = int(chunk_duration * self._sample_rate)
        max_samples = int(max_duration * self._sample_rate)
        silence_samples = int(silence_duration * self._sample_rate)

        logger.info(
            f"Recording until silence (threshold={silence_threshold}, "
            f"max={max_duration}s)..."
        )

        chunks: list[np.ndarray] = []
        silent_samples_count = 0

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
        ) as stream:
            total_samples = 0
            while total_samples < max_samples:
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.squeeze()
                chunks.append(chunk)
                total_samples += len(chunk)

                # Check RMS level
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms < silence_threshold:
                    silent_samples_count += len(chunk)
                else:
                    silent_samples_count = 0

                if silent_samples_count >= silence_samples:
                    logger.info("Silence detected, stopping recording")
                    break

        audio = np.concatenate(chunks)
        logger.info(f"Recording complete: {len(audio)} samples ({len(audio) / self._sample_rate:.1f}s)")
        return audio

    def save(self, audio: np.ndarray, path: str) -> str:
        """
        Save audio data to a WAV file.

        Args:
            audio: Numpy array with float32 audio data.
            path: Output file path.

        Returns:
            The path where the file was saved.
        """
        import soundfile as sf

        sf.write(path, audio, self._sample_rate)
        logger.debug(f"Audio saved to {path}")
        return path

    def save_temp(self, audio: np.ndarray) -> str:
        """
        Save audio data to a temporary WAV file.

        Args:
            audio: Numpy array with float32 audio data.

        Returns:
            Path to the temporary WAV file.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        return self.save(audio, tmp.name)

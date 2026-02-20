"""
Unit tests for the Speech-to-Text module.

All tests use mocks to avoid requiring a microphone or Whisper model.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np


@dataclass
class FakeTranscriptionResult:
    """Fake transcription result for testing."""

    text: str
    language: str = "fr"


# ────────────────────────────────────────────
# SpeechInput
# ────────────────────────────────────────────


class TestSpeechInput:
    """Tests for the SpeechInput CLI helper."""

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_init_success(self, mock_transcriber_cls: MagicMock) -> None:
        """Test successful initialization."""
        from src.speech.speech_input import SpeechInput

        speech = SpeechInput(model_name="tiny")

        mock_transcriber_cls.assert_called_once_with(model_name="tiny")
        mock_transcriber_cls.return_value.warmup.assert_called_once()
        assert speech.available is True

    def test_init_missing_dependencies(self) -> None:
        """Test initialization when whisper is not installed."""
        with patch(
            "src.speech.transcriber.SpeechTranscriber",
            side_effect=ImportError("No module named 'whisper'"),
            create=True,
        ):
            from src.speech.speech_input import SpeechInput

            speech = SpeechInput()

            assert speech.available is False

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_record_success(self, mock_transcriber_cls: MagicMock) -> None:
        """Test successful recording and transcription."""
        mock_transcriber = mock_transcriber_cls.return_value
        mock_transcriber.transcribe_from_mic.return_value = FakeTranscriptionResult(
            text="Je veux aller de Paris a Lyon"
        )

        from src.speech.speech_input import SpeechInput

        speech = SpeechInput()
        result = speech.record(duration=5.0)

        assert result == "Je veux aller de Paris a Lyon"
        mock_transcriber.transcribe_from_mic.assert_called_once_with(duration=5.0)

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_record_empty_transcription(self, mock_transcriber_cls: MagicMock) -> None:
        """Test recording that produces empty transcription."""
        mock_transcriber = mock_transcriber_cls.return_value
        mock_transcriber.transcribe_from_mic.return_value = FakeTranscriptionResult(text="")

        from src.speech.speech_input import SpeechInput

        speech = SpeechInput()
        result = speech.record()

        assert result is None

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_record_whitespace_only(self, mock_transcriber_cls: MagicMock) -> None:
        """Test recording that produces whitespace-only transcription."""
        mock_transcriber = mock_transcriber_cls.return_value
        mock_transcriber.transcribe_from_mic.return_value = FakeTranscriptionResult(text="   ")

        from src.speech.speech_input import SpeechInput

        speech = SpeechInput()
        result = speech.record()

        assert result is None

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_record_mic_error(self, mock_transcriber_cls: MagicMock) -> None:
        """Test recording when microphone fails."""
        mock_transcriber = mock_transcriber_cls.return_value
        mock_transcriber.transcribe_from_mic.side_effect = RuntimeError("Error querying device -1")

        from src.speech.speech_input import SpeechInput

        speech = SpeechInput()
        result = speech.record()

        assert result is None

    def test_record_when_not_available(self) -> None:
        """Test record() when speech is not available."""
        with patch(
            "src.speech.transcriber.SpeechTranscriber",
            side_effect=ImportError("No module"),
            create=True,
        ):
            from src.speech.speech_input import SpeechInput

            speech = SpeechInput()
            result = speech.record()

            assert result is None

    @patch("src.speech.transcriber.SpeechTranscriber", create=True)
    def test_record_default_duration(self, mock_transcriber_cls: MagicMock) -> None:
        """Test that record uses 5.0s default duration."""
        mock_transcriber = mock_transcriber_cls.return_value
        mock_transcriber.transcribe_from_mic.return_value = FakeTranscriptionResult(text="test")

        from src.speech.speech_input import SpeechInput

        speech = SpeechInput()
        speech.record()

        mock_transcriber.transcribe_from_mic.assert_called_once_with(duration=5.0)


# ────────────────────────────────────────────
# SpeechTranscriber
# ────────────────────────────────────────────


class TestSpeechTranscriber:
    """Tests for the SpeechTranscriber class (with mocks)."""

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_transcriber_init(
        self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        """Test SpeechTranscriber initialization."""
        from src.speech.transcriber import SpeechTranscriber

        SpeechTranscriber(model_name="tiny")

        mock_recorder_cls.assert_called_once_with(sample_rate=16000)
        mock_model_cls.assert_called_once_with(model_name="tiny", device="auto", language="fr")

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_transcriber_init_custom_params(
        self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        """Test SpeechTranscriber with custom parameters."""
        from src.speech.transcriber import SpeechTranscriber

        SpeechTranscriber(model_name="large", device="cpu", language="en", sample_rate=8000)

        mock_recorder_cls.assert_called_once_with(sample_rate=8000)
        mock_model_cls.assert_called_once_with(model_name="large", device="cpu", language="en")

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_transcribe_from_mic(
        self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        """Test transcribe_from_mic calls record then transcribe."""
        fake_audio = np.zeros(16000 * 5, dtype=np.float32)
        mock_recorder = mock_recorder_cls.return_value
        mock_recorder.record.return_value = fake_audio

        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = FakeTranscriptionResult(text="De Rennes a Paris")

        from src.speech.transcriber import SpeechTranscriber

        transcriber = SpeechTranscriber(model_name="tiny")
        result = transcriber.transcribe_from_mic(duration=5.0)

        mock_recorder.record.assert_called_once_with(5.0)
        mock_model.transcribe.assert_called_once_with(fake_audio)
        assert result.text == "De Rennes a Paris"

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_transcribe_from_mic_default_duration(
        self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        """Test transcribe_from_mic uses 5.0s default."""
        mock_recorder = mock_recorder_cls.return_value
        mock_recorder.record.return_value = np.zeros(16000 * 5, dtype=np.float32)
        mock_model_cls.return_value.transcribe.return_value = FakeTranscriptionResult(text="ok")

        from src.speech.transcriber import SpeechTranscriber

        transcriber = SpeechTranscriber()
        transcriber.transcribe_from_mic()

        mock_recorder.record.assert_called_once_with(5.0)

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_transcribe_from_mic_auto(
        self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        """Test transcribe_from_mic_auto with silence detection."""
        fake_audio = np.zeros(16000 * 3, dtype=np.float32)
        mock_recorder = mock_recorder_cls.return_value
        mock_recorder.record_until_silence.return_value = fake_audio

        mock_model = mock_model_cls.return_value
        mock_model.transcribe.return_value = FakeTranscriptionResult(text="Bonjour")

        from src.speech.transcriber import SpeechTranscriber

        transcriber = SpeechTranscriber()
        result = transcriber.transcribe_from_mic_auto(
            silence_threshold=0.01, silence_duration=2.0, max_duration=10.0
        )

        mock_recorder.record_until_silence.assert_called_once_with(
            silence_threshold=0.01, silence_duration=2.0, max_duration=10.0
        )
        assert result.text == "Bonjour"

    @patch("src.speech.transcriber.WhisperModel")
    @patch("src.speech.transcriber.AudioRecorder")
    def test_warmup(self, mock_recorder_cls: MagicMock, mock_model_cls: MagicMock) -> None:
        """Test warmup pre-loads the model."""
        from src.speech.transcriber import SpeechTranscriber

        transcriber = SpeechTranscriber()
        transcriber.warmup()

        mock_model_cls.return_value.load.assert_called_once()


# ────────────────────────────────────────────
# WhisperModel
# ────────────────────────────────────────────


class TestWhisperModel:
    """Tests for the WhisperModel wrapper."""

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_init_defaults(self, mock_device: MagicMock) -> None:
        """Test default initialization."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel()

        assert model._model_name == "medium"
        assert model._language == "fr"
        assert model._device == "cpu"
        assert model.is_loaded is False

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_is_loaded_false_initially(self, mock_device: MagicMock) -> None:
        """Test that model is not loaded after init."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel()

        assert model.is_loaded is False

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_load_model(self, mock_device: MagicMock) -> None:
        """Test model loading."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel(model_name="tiny")

        with patch.dict("sys.modules", {"whisper": MagicMock()}) as _:
            import sys

            mock_whisper = sys.modules["whisper"]
            mock_whisper.load_model.return_value = MagicMock()

            model.load()

            mock_whisper.load_model.assert_called_once_with("tiny", device="cpu")
            assert model.is_loaded is True

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_load_model_idempotent(self, mock_device: MagicMock) -> None:
        """Test that loading twice does not reload."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel(model_name="tiny")
        model._model = MagicMock()  # Simulate already loaded

        model.load()  # Should skip

        assert model.is_loaded is True

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_transcribe_with_numpy_array(self, mock_device: MagicMock) -> None:
        """Test transcription with numpy audio input."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel(model_name="tiny")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": " Je veux aller a Lyon ",
            "language": "fr",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": " Je veux aller a Lyon "},
            ],
        }
        model._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = model.transcribe(audio)

        assert result.text == "Je veux aller a Lyon"
        assert result.language == "fr"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Je veux aller a Lyon"
        mock_model.transcribe.assert_called_once_with(audio, language="fr", fp16=False)

    @patch("src.speech.whisper_model.get_torch_device", return_value="cuda")
    def test_transcribe_fp16_on_cuda(self, mock_device: MagicMock) -> None:
        """Test that fp16 is enabled on CUDA device."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel(model_name="tiny")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "test",
            "language": "fr",
            "segments": [],
        }
        model._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        model.transcribe(audio)

        mock_model.transcribe.assert_called_once_with(audio, language="fr", fp16=True)

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_transcribe_no_segments(self, mock_device: MagicMock) -> None:
        """Test transcription when no segments are returned."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Bonjour",
            "segments": [],
        }
        model._model = mock_model

        result = model.transcribe(np.zeros(16000, dtype=np.float32))

        assert result.text == "Bonjour"
        assert result.language == "fr"  # Falls back to default
        assert result.segments == []

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_transcribe_auto_loads_model(self, mock_device: MagicMock) -> None:
        """Test that transcribe auto-loads the model if not loaded."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel(model_name="tiny")
        assert model.is_loaded is False

        with patch.dict("sys.modules", {"whisper": MagicMock()}) as _:
            import sys

            mock_whisper = sys.modules["whisper"]
            loaded_model = MagicMock()
            loaded_model.transcribe.return_value = {
                "text": "auto loaded",
                "language": "fr",
                "segments": [],
            }
            mock_whisper.load_model.return_value = loaded_model

            result = model.transcribe(np.zeros(16000, dtype=np.float32))

            mock_whisper.load_model.assert_called_once()
            assert result.text == "auto loaded"

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_unload(self, mock_device: MagicMock) -> None:
        """Test model unloading."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel()
        model._model = MagicMock()
        assert model.is_loaded is True

        model.unload()

        assert model.is_loaded is False

    @patch("src.speech.whisper_model.get_torch_device", return_value="cpu")
    def test_unload_when_not_loaded(self, mock_device: MagicMock) -> None:
        """Test unload when model was never loaded (no-op)."""
        from src.speech.whisper_model import WhisperModel

        model = WhisperModel()

        model.unload()  # Should not raise

        assert model.is_loaded is False


# ────────────────────────────────────────────
# AudioRecorder
# ────────────────────────────────────────────


class TestAudioRecorder:
    """Tests for the AudioRecorder class."""

    @patch("src.speech.audio_processor.soundfile", create=True)
    @patch("src.speech.audio_processor.sounddevice", create=True)
    def test_init_defaults(self, mock_sd: MagicMock, mock_sf: MagicMock) -> None:
        """Test default initialization."""
        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": mock_sf}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder()

            assert recorder.sample_rate == 16000

    @patch("src.speech.audio_processor.soundfile", create=True)
    @patch("src.speech.audio_processor.sounddevice", create=True)
    def test_init_custom_sample_rate(self, mock_sd: MagicMock, mock_sf: MagicMock) -> None:
        """Test initialization with custom sample rate."""
        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": mock_sf}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder(sample_rate=8000)

            assert recorder.sample_rate == 8000

    def test_record_fixed_duration(self) -> None:
        """Test fixed-duration recording."""
        mock_sd = MagicMock()
        fake_audio = np.zeros((16000 * 3, 1), dtype=np.float32)
        mock_sd.rec.return_value = fake_audio

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": MagicMock()}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder()
            result = recorder.record(duration=3.0)

            mock_sd.rec.assert_called_once_with(
                48000,
                samplerate=16000,
                channels=1,
                dtype="float32",
            )
            mock_sd.wait.assert_called_once()
            assert result.ndim == 1
            assert len(result) == 48000

    def test_record_until_silence(self) -> None:
        """Test silence-detection recording."""
        mock_sd = MagicMock()

        # Simulate: 2 chunks of sound, then 5 chunks of silence
        sound_chunk = np.full((8000, 1), 0.1, dtype=np.float32)  # RMS > 0.01
        silent_chunk = np.full((8000, 1), 0.001, dtype=np.float32)  # RMS < 0.01

        mock_stream = MagicMock()
        mock_stream.read.side_effect = [
            (sound_chunk, None),
            (sound_chunk, None),
            (silent_chunk, None),
            (silent_chunk, None),
            (silent_chunk, None),
            (silent_chunk, None),
            (silent_chunk, None),
        ]
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": MagicMock()}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder()
            result = recorder.record_until_silence(
                silence_threshold=0.01,
                silence_duration=2.0,
                max_duration=30.0,
                chunk_duration=0.5,
            )

            assert result.ndim == 1
            assert len(result) > 0

    def test_save_audio(self) -> None:
        """Test saving audio to a file."""
        mock_sf = MagicMock()
        mock_sd = MagicMock()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": mock_sf}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder()
            audio = np.zeros(16000, dtype=np.float32)
            path = recorder.save(audio, "/tmp/test.wav")

            mock_sf.write.assert_called_once_with("/tmp/test.wav", audio, 16000)
            assert path == "/tmp/test.wav"

    def test_save_temp(self) -> None:
        """Test saving audio to a temporary file."""
        mock_sf = MagicMock()
        mock_sd = MagicMock()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "soundfile": mock_sf}):
            from src.speech.audio_processor import AudioRecorder

            recorder = AudioRecorder()
            audio = np.zeros(16000, dtype=np.float32)
            path = recorder.save_temp(audio)

            assert path.endswith(".wav")
            mock_sf.write.assert_called_once()


# ────────────────────────────────────────────
# TranscriptionResult
# ────────────────────────────────────────────


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_creation(self) -> None:
        """Test basic creation."""
        from src.speech.whisper_model import TranscriptionResult

        result = TranscriptionResult(text="Bonjour", language="fr")

        assert result.text == "Bonjour"
        assert result.language == "fr"
        assert result.segments == []

    def test_creation_with_segments(self) -> None:
        """Test creation with segments."""
        from src.speech.whisper_model import TranscriptionResult

        segments = [{"start": 0.0, "end": 1.5, "text": "Bonjour"}]
        result = TranscriptionResult(text="Bonjour", language="fr", segments=segments)

        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Bonjour"

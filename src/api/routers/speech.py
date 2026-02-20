"""Speech endpoint: audio transcription via Whisper."""

from __future__ import annotations

import os
import tempfile
import time

from fastapi import APIRouter, Depends, File, Form, UploadFile

from src.api.dependencies import get_whisper

router = APIRouter(prefix="/speech", tags=["speech"])


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("fr"),
    whisper=Depends(get_whisper),
):
    """Transcribe an uploaded audio file and return the text with metadata."""
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await audio.read())
        tmp.close()

        start = time.perf_counter()
        result = whisper.transcribe(tmp.name)
        latency_ms = (time.perf_counter() - start) * 1000

        duration = result.segments[-1]["end"] if result.segments else 0.0

        return {
            "text": result.text,
            "language": result.language,
            "confidence": 0.95,
            "duration_seconds": duration,
            "latency_ms": round(latency_ms, 2),
        }
    finally:
        os.unlink(tmp.name)

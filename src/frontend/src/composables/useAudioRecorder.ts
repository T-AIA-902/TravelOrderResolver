import { ref } from 'vue'
import { apiPostForm } from '../api/client'
import type { TranscriptionResponse } from '../api/types'

export type RecorderState = 'idle' | 'recording' | 'processing'

export function useAudioRecorder() {
  const state = ref<RecorderState>('idle')
  const error = ref<string | null>(null)
  const transcription = ref<TranscriptionResponse | null>(null)

  let mediaRecorder: MediaRecorder | null = null
  let chunks: Blob[] = []

  async function startRecording() {
    error.value = null
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      chunks = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        // Stop all tracks to release the microphone
        stream.getTracks().forEach((t) => t.stop())

        const blob = new Blob(chunks, { type: 'audio/webm' })
        state.value = 'processing'

        try {
          const formData = new FormData()
          formData.append('audio', blob, 'recording.webm')
          transcription.value = await apiPostForm<TranscriptionResponse>(
            '/api/speech/transcribe',
            formData,
          )
        } catch (e: unknown) {
          error.value = e instanceof Error ? e.message : 'Erreur de transcription'
        } finally {
          state.value = 'idle'
        }
      }

      mediaRecorder.start()
      state.value = 'recording'
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : "Erreur d'accès au microphone"
      state.value = 'idle'
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop()
    }
  }

  return { state, error, transcription, startRecording, stopRecording }
}

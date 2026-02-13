import { apiPostForm } from './client'
import type { TranscriptionResponse } from './types'

export function transcribeAudio(audioBlob: Blob, language?: string) {
  const formData = new FormData()
  formData.append('audio', audioBlob, 'recording.webm')
  if (language) formData.append('language', language)
  return apiPostForm<TranscriptionResponse>('/api/speech/transcribe', formData)
}

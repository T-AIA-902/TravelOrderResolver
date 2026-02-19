import { apiPostForm } from './client'
import type { TranscriptionResponse } from './types'

export function transcribe(audioBlob: Blob, language?: string): Promise<TranscriptionResponse> {
  const formData = new FormData()
  formData.append('audio', audioBlob, 'recording.webm')
  if (language) formData.append('language', language)
  return apiPostForm<TranscriptionResponse>('/api/speech/transcribe', formData)
}

/** @deprecated Use `transcribe` instead */
export const transcribeAudio = transcribe

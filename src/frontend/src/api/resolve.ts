import { apiPost } from './client'
import type {
  ResolveRequest,
  ResolveResponse,
  NlpLanguageItem,
  NlpIntentItem,
  NlpEntityItem,
} from './types'

export function resolve(req: ResolveRequest) {
  return apiPost<ResolveResponse>('/api/resolve', req)
}

export function detectLanguage(text: string) {
  return apiPost<{ results: NlpLanguageItem[] }>('/api/nlp/language', { text })
}

export function classifyIntent(text: string) {
  return apiPost<{ results: NlpIntentItem[] }>('/api/nlp/intent', { text })
}

export function extractEntities(text: string) {
  return apiPost<{ results: NlpEntityItem[] }>('/api/nlp/entities', { text })
}

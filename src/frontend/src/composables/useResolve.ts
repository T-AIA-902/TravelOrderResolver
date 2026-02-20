import { ref } from 'vue'
import { resolve } from '../api/resolve'
import type { ResolveRequest, ResolveResponse } from '../api/types'

export function useResolve() {
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const lastResponse = ref<ResolveResponse | null>(null)

  async function resolveTrip(request: ResolveRequest): Promise<ResolveResponse> {
    isLoading.value = true
    error.value = null
    try {
      const result = await resolve(request)
      lastResponse.value = result
      return result
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : 'Erreur lors de la r\u00e9solution'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  return { isLoading, error, lastResponse, resolveTrip }
}

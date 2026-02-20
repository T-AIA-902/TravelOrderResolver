import { ref, onMounted, onUnmounted } from 'vue'
import type { RequestMetric, ResourceSnapshot, CarbonSummary } from '../api/types'
import { getMetrics, getResources, getCarbonSummary } from '../api/monitoring'

export function useMonitoring(pollInterval = 5000) {
  const metrics = ref<RequestMetric[]>([])
  const totalRequests = ref(0)
  const avgLatency = ref(0)
  const resources = ref<ResourceSnapshot | null>(null)
  const carbon = ref<CarbonSummary | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  let timer: ReturnType<typeof setInterval> | null = null

  async function fetchAll() {
    isLoading.value = true
    error.value = null

    try {
      const [metricsRes, resourcesRes, carbonRes] = await Promise.allSettled([
        getMetrics(),
        getResources(),
        getCarbonSummary(),
      ])

      if (metricsRes.status === 'fulfilled') {
        metrics.value = metricsRes.value.requests
        totalRequests.value = metricsRes.value.total_requests
        avgLatency.value = metricsRes.value.avg_latency_ms
      }

      if (resourcesRes.status === 'fulfilled') {
        resources.value = resourcesRes.value
      }

      if (carbonRes.status === 'fulfilled') {
        carbon.value = carbonRes.value
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Erreur monitoring'
    } finally {
      isLoading.value = false
    }
  }

  function startPolling() {
    fetchAll()
    timer = setInterval(fetchAll, pollInterval)
  }

  function stopPolling() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  onMounted(startPolling)
  onUnmounted(stopPolling)

  return {
    metrics,
    totalRequests,
    avgLatency,
    resources,
    carbon,
    isLoading,
    error,
    refresh: fetchAll,
  }
}

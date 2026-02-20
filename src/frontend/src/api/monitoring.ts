import { apiGet } from './client'
import type { MonitoringMetricsResponse, ResourceSnapshot, CarbonSummary } from './types'

export function getMetrics(limit = 100) {
  return apiGet<MonitoringMetricsResponse>(`/api/monitoring/metrics?limit=${limit}`)
}

export function getResources() {
  return apiGet<ResourceSnapshot>('/api/monitoring/resources')
}

export function getCarbonSummary() {
  return apiGet<CarbonSummary>('/api/monitoring/carbon')
}

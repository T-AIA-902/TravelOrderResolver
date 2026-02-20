import { apiGet, apiPost } from './client'
import type {
  EvalReport,
  EvalReportDetail,
  EvalRunRequest,
  EvalRunResponse,
  EvalStatus,
} from './types'

export function listReports() {
  return apiGet<{ reports: EvalReport[] }>('/api/evaluation/reports')
}

export function getReport(reportId: string) {
  return apiGet<EvalReportDetail>(`/api/evaluation/reports/${encodeURIComponent(reportId)}`)
}

export function runEvaluation(req: EvalRunRequest) {
  return apiPost<EvalRunResponse>('/api/evaluation/run', req)
}

export function getEvalStatus(taskId: string) {
  return apiGet<EvalStatus>(`/api/evaluation/status/${encodeURIComponent(taskId)}`)
}

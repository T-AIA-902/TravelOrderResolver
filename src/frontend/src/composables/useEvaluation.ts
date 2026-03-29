import { ref, onMounted, onUnmounted } from 'vue'
import { listReports, getReport, runEvaluation, getEvalStatus } from '../api/evaluation'
import type {
  EvalReport,
  EvalReportDetail,
  EvalRunRequest,
  EvalRunResponse,
  EvalStatus,
} from '../api/types'

const TASK_ID_KEY = 'eval_task_id'

export function useEvaluation() {
  const reports = ref<EvalReport[]>([])
  const reportsLoading = ref(false)
  const reportsError = ref<string | null>(null)

  const reportDetail = ref<EvalReportDetail | null>(null)
  const detailLoading = ref(false)
  const detailError = ref<string | null>(null)

  const runLoading = ref(false)
  const runError = ref<string | null>(null)

  const evalStatus = ref<EvalStatus | null>(null)
  let pollingTimer: ReturnType<typeof setInterval> | null = null

  async function loadReports() {
    reportsLoading.value = true
    reportsError.value = null
    try {
      const data = await listReports()
      reports.value = data.reports
    } catch (e: unknown) {
      reportsError.value = e instanceof Error ? e.message : 'Erreur lors du chargement des rapports'
    } finally {
      reportsLoading.value = false
    }
  }

  async function loadReport(reportId: string) {
    detailLoading.value = true
    detailError.value = null
    try {
      reportDetail.value = await getReport(reportId)
    } catch (e: unknown) {
      detailError.value = e instanceof Error ? e.message : 'Erreur lors du chargement du rapport'
    } finally {
      detailLoading.value = false
    }
  }

  async function loadLatestReport() {
    const latest = reports.value[0]
    if (latest) {
      await loadReport(latest.id)
    }
  }

  async function startEvaluation(request: EvalRunRequest): Promise<EvalRunResponse | null> {
    runLoading.value = true
    runError.value = null
    evalStatus.value = null
    try {
      const response = await runEvaluation(request)
      // Persist task_id so we can resume polling after page navigation
      localStorage.setItem(TASK_ID_KEY, response.task_id)
      startPolling(response.task_id)
      return response
    } catch (e: unknown) {
      runError.value = e instanceof Error ? e.message : "Erreur lors du lancement de l'évaluation"
      return null
    } finally {
      runLoading.value = false
    }
  }

  function startPolling(taskId: string) {
    stopPolling()
    pollingTimer = setInterval(async () => {
      try {
        evalStatus.value = await getEvalStatus(taskId)
        if (evalStatus.value.status === 'completed' || evalStatus.value.status === 'failed') {
          stopPolling()
          localStorage.removeItem(TASK_ID_KEY)
          if (evalStatus.value.status === 'completed') {
            await loadReports()
            await loadLatestReport()
          }
        }
      } catch {
        stopPolling()
        localStorage.removeItem(TASK_ID_KEY)
      }
    }, 2000)
  }

  function stopPolling() {
    if (pollingTimer !== null) {
      clearInterval(pollingTimer)
      pollingTimer = null
    }
  }

  /** Resume polling if an evaluation was running before page navigation. */
  async function resumeIfRunning() {
    const savedTaskId = localStorage.getItem(TASK_ID_KEY)
    if (!savedTaskId) return

    try {
      const status = await getEvalStatus(savedTaskId)
      evalStatus.value = status
      if (status.status === 'running' || status.status === 'started') {
        startPolling(savedTaskId)
      } else {
        // Task finished while we were away
        localStorage.removeItem(TASK_ID_KEY)
        if (status.status === 'completed') {
          await loadReports()
          await loadLatestReport()
        }
      }
    } catch {
      // Task ID no longer valid (server restarted, etc.)
      localStorage.removeItem(TASK_ID_KEY)
    }
  }

  onMounted(() => {
    resumeIfRunning()
  })

  onUnmounted(() => {
    // Stop polling but DON'T clear the task_id — we want to resume later
    stopPolling()
  })

  return {
    reports,
    reportsLoading,
    reportsError,
    reportDetail,
    detailLoading,
    detailError,
    runLoading,
    runError,
    evalStatus,
    loadReports,
    loadReport,
    loadLatestReport,
    startEvaluation,
    stopPolling,
  }
}

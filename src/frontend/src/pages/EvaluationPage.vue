<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Play, BarChart3 } from 'lucide-vue-next'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import MetricsTable from '../components/evaluation/MetricsTable.vue'
import ConfusionMatrixView from '../components/evaluation/ConfusionMatrixView.vue'
import EvalProgress from '../components/evaluation/EvalProgress.vue'
import { useEvaluation } from '../composables/useEvaluation'
import type { ModelEvalResult } from '../api/types'

const {
  reportDetail,
  detailLoading,
  detailError,
  runLoading,
  runError,
  evalStatus,
  loadReports,
  loadLatestReport,
  startEvaluation,
} = useEvaluation()

// --- Results ---
const activeTab = ref<'language' | 'intent' | 'entity' | 'entity_fuzzy'>('language')

const tabs = [
  { key: 'language' as const, label: 'Langue' },
  { key: 'intent' as const, label: 'Intent' },
  { key: 'entity' as const, label: 'Entités' },
  { key: 'entity_fuzzy' as const, label: 'Entités + Fuzzy' },
]

const currentReport = computed(() => reportDetail.value)

const activeResults = computed<Record<string, ModelEvalResult> | undefined>(() => {
  return currentReport.value?.[activeTab.value]
})

const hasConfusionMatrix = computed(() => {
  if (!activeResults.value) return false
  return Object.values(activeResults.value).some((r) => r.confusion_matrix)
})

// --- Run ---
async function handleRun() {
  await startEvaluation({ eval_type: 'all' })
}

const isEvalRunning = computed(
  () =>
    runLoading.value ||
    evalStatus.value?.status === 'running' ||
    evalStatus.value?.status === 'started',
)

// --- Init ---
onMounted(async () => {
  await loadReports()
  await loadLatestReport()
})
</script>

<template>
  <div class="space-y-6 p-8">
    <!-- Header -->
    <div class="flex items-center gap-4">
      <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-50">
        <BarChart3 class="h-6 w-6 text-violet-600" />
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-gray-900">Évaluation</h1>
        <p class="text-gray-500">Métriques d'évaluation et lancement de benchmarks</p>
      </div>
    </div>

    <!-- ===================== Évaluation du modèle ===================== -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center justify-between border-b border-gray-200 px-6 py-5">
        <h2 class="text-lg font-semibold text-gray-900">Évaluation du modèle</h2>
        <button
          type="button"
          class="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
          :disabled="isEvalRunning"
          @click="handleRun"
        >
          <LoadingSpinner v-if="isEvalRunning" size="sm" />
          <Play v-else class="h-4 w-4" />
          {{ isEvalRunning ? 'En cours...' : 'Lancer' }}
        </button>
      </div>

      <div class="px-6 py-5">
        <ErrorAlert v-if="runError" :message="runError" class="mb-4" />

        <!-- Progress -->
        <EvalProgress v-if="isEvalRunning" :status="evalStatus" class="mb-4" />

        <!-- Loading results -->
        <div v-if="detailLoading" class="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </div>

        <ErrorAlert v-else-if="detailError" :message="detailError" />

        <!-- Results -->
        <div v-else>
          <!-- Tabs -->
          <div class="mb-5 flex gap-1 rounded-lg bg-gray-100 p-1">
            <button
              v-for="tab in tabs"
              :key="tab.key"
              class="rounded-md px-5 py-2.5 text-sm font-medium transition"
              :class="
                activeTab === tab.key
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              "
              @click="activeTab = tab.key"
            >
              {{ tab.label }}
            </button>
          </div>

          <!-- Tab content -->
          <div v-if="activeResults && Object.keys(activeResults).length > 0">
            <MetricsTable :model-results="activeResults" />

            <div v-if="hasConfusionMatrix" class="mt-6 space-y-4">
              <h3 class="text-sm font-semibold text-gray-900">Matrices de confusion</h3>
              <div class="grid grid-cols-1 gap-6 xl:grid-cols-2">
                <div v-for="(result, model) in activeResults" :key="'cm-' + String(model)">
                  <template v-if="result.confusion_matrix">
                    <p class="mb-2 text-sm font-medium text-gray-700">{{ model }}</p>
                    <ConfusionMatrixView :matrix="result.confusion_matrix" />
                  </template>
                </div>
              </div>
            </div>
          </div>

          <div v-else class="rounded-lg bg-gray-50 py-12 text-center text-sm text-gray-400">
            Aucune donnée disponible pour cet onglet.
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

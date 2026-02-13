<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Play } from 'lucide-vue-next'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import MetricsTable from '../components/evaluation/MetricsTable.vue'
import ConfusionMatrixView from '../components/evaluation/ConfusionMatrixView.vue'
import EvalProgress from '../components/evaluation/EvalProgress.vue'
import { useEvaluation } from '../composables/useEvaluation'
import type { EvalReportDetail, ModelEvalResult } from '../api/types'

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

// --- Mock data (à supprimer quand le backend est prêt) ---
const MOCK_REPORT: EvalReportDetail = {
  id: 'mock',
  date: new Date().toISOString(),
  config: {
    eval_type: 'all',
    dataset: 'eval_dataset.csv',
    samples: 200,
    device: 'cpu',
    preprocess: true,
  },
  language: {
    CamemBERT: {
      accuracy: 0.96,
      per_class: {
        FRENCH: { precision: 0.97, recall: 0.98, f1: 0.975, support: 150 },
        ENGLISH: { precision: 0.94, recall: 0.92, f1: 0.93, support: 40 },
        UNKNOWN: { precision: 0.85, recall: 0.8, f1: 0.824, support: 10 },
      },
      avg_latency_ms: 12.3,
      confusion_matrix: {
        labels: ['FRENCH', 'ENGLISH', 'UNKNOWN'],
        matrix: [
          [147, 2, 1],
          [1, 37, 2],
          [1, 1, 8],
        ],
      },
    },
  },
  intent: {
    CamemBERT: {
      accuracy: 0.92,
      per_class: {
        TRIP: { precision: 0.95, recall: 0.93, f1: 0.94, support: 120 },
        GREETING: { precision: 0.88, recall: 0.9, f1: 0.89, support: 30 },
        OTHER: { precision: 0.85, recall: 0.88, f1: 0.865, support: 50 },
      },
      avg_latency_ms: 18.7,
      confusion_matrix: {
        labels: ['TRIP', 'GREETING', 'OTHER'],
        matrix: [
          [112, 3, 5],
          [1, 27, 2],
          [2, 4, 44],
        ],
      },
    },
  },
  entity: {
    CamemBERT: {
      accuracy: 0.89,
      per_class: {
        departure: { precision: 0.91, recall: 0.88, f1: 0.895, support: 120 },
        destination: { precision: 0.9, recall: 0.87, f1: 0.885, support: 120 },
        intermediate: { precision: 0.82, recall: 0.78, f1: 0.8, support: 45 },
      },
      avg_latency_ms: 22.1,
    },
  },
  entity_fuzzy: {
    CamemBERT: {
      accuracy: 0.93,
      per_class: {
        departure: { precision: 0.95, recall: 0.93, f1: 0.94, support: 120 },
        destination: { precision: 0.94, recall: 0.92, f1: 0.93, support: 120 },
        intermediate: { precision: 0.88, recall: 0.85, f1: 0.865, support: 45 },
      },
      avg_latency_ms: 24.5,
    },
  },
}

// --- Results ---
const activeTab = ref<'language' | 'intent' | 'entity' | 'entity_fuzzy'>('language')

const tabs = [
  { key: 'language' as const, label: 'Langue' },
  { key: 'intent' as const, label: 'Intent' },
  { key: 'entity' as const, label: 'Entités' },
  { key: 'entity_fuzzy' as const, label: 'Entités + Fuzzy' },
]

const currentReport = computed(() => reportDetail.value ?? MOCK_REPORT)

const activeResults = computed<Record<string, ModelEvalResult> | undefined>(() => {
  return currentReport.value[activeTab.value]
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
  <div class="space-y-8 p-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Évaluation</h1>
      <p class="text-gray-600">Métriques d'évaluation et lancement de benchmarks</p>
    </div>

    <!-- ===================== Évaluation du modèle ===================== -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center justify-between border-b border-gray-200 px-5 py-4">
        <h2 class="text-lg font-semibold text-gray-900">Évaluation du modèle</h2>
        <button
          type="button"
          class="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
          :disabled="isEvalRunning"
          @click="handleRun"
        >
          <LoadingSpinner v-if="isEvalRunning" size="sm" />
          <Play v-else class="h-4 w-4" />
          {{ isEvalRunning ? 'En cours...' : 'Lancer' }}
        </button>
      </div>

      <div class="px-5 py-4">
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
          <div class="mb-4 flex gap-1 rounded-lg bg-gray-100 p-1">
            <button
              v-for="tab in tabs"
              :key="tab.key"
              class="rounded-md px-4 py-2 text-sm font-medium transition"
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

          <div v-else class="py-8 text-center text-sm text-gray-500">
            Aucune donnée disponible pour cet onglet.
          </div>
        </div>
      </div>
    </div>

    <!-- ===================== Comparaison des modèles ===================== -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="border-b border-gray-200 px-5 py-4">
        <h2 class="text-lg font-semibold text-gray-900">Comparaison des modèles</h2>
      </div>

      <div class="px-5 py-8 text-center text-sm text-gray-500">
        Les résultats comparatifs des autres modèles seront ajoutés ici.
      </div>
    </div>
  </div>
</template>

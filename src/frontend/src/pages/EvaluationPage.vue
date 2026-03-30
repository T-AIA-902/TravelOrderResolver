<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { Play, BarChart3, ChevronDown, Download, Trophy, Zap, Target, Languages } from 'lucide-vue-next'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import EvalProgress from '../components/evaluation/EvalProgress.vue'
import { useEvaluation } from '../composables/useEvaluation'

const {
  reports,
  reportDetail,
  reportsLoading,
  detailLoading,
  detailError,
  runLoading,
  runError,
  evalStatus,
  loadReports,
  loadReport,
  startEvaluation,
} = useEvaluation()

// --- Report selection ---
const selectedReportId = ref<string | null>(null)

watch(selectedReportId, async (id) => {
  if (id) await loadReport(id)
})

// --- Tabs ---
type TabKey = 'language' | 'intent' | 'entity' | 'entity_fuzzy' | 'combined' | 'combined_fuzzy'
const activeTab = ref<TabKey>('intent')

const tabs: { key: TabKey; label: string }[] = [
  { key: 'language', label: 'Langue' },
  { key: 'intent', label: 'Intent' },
  { key: 'entity', label: 'Entités' },
  { key: 'entity_fuzzy', label: 'Entités + Fuzzy' },
  { key: 'combined', label: 'Combiné' },
  { key: 'combined_fuzzy', label: 'Combiné + Fuzzy' },
]

// --- Results ---
type MetricRow = Record<string, number>
type SectionData = Record<string, MetricRow>

const activeResults = computed((): SectionData | null => {
  if (!reportDetail.value) return null
  const data = reportDetail.value as Record<string, unknown>
  const section = data[activeTab.value]

  // combined/combined_fuzzy are arrays, convert to dict
  if (Array.isArray(section)) {
    const dict: SectionData = {}
    for (const item of section) {
      const key = `${item.intent} + ${item.entity}`
      dict[key] = { ...item }
      delete dict[key].intent
      delete dict[key].entity
      delete dict[key].with_fuzzy
    }
    return dict
  }

  return (section as SectionData) || null
})

// --- Summary cards ---
interface SummaryCard {
  label: string
  model: string
  value: string
  icon: typeof Trophy
  color: string
}

const summaryCards = computed((): SummaryCard[] => {
  if (!reportDetail.value) return []
  const data = reportDetail.value as Record<string, unknown>
  const cards: SummaryCard[] = []

  // Best intent
  const intent = data.intent as SectionData | undefined
  if (intent) {
    const best = findBest(intent, 'accuracy')
    if (best) cards.push({ label: 'Meilleur Intent', model: best.model, value: pct(best.value), icon: Zap, color: 'blue' })
  }

  // Best entity+fuzzy
  const entityFuzzy = data.entity_fuzzy as SectionData | undefined
  if (entityFuzzy) {
    const best = findBest(entityFuzzy, 'macro_f1')
    if (best) cards.push({ label: 'Meilleur Entity (Fuzzy)', model: best.model, value: pct(best.value), icon: Target, color: 'green' })
  }

  // Best entity raw
  const entity = data.entity as SectionData | undefined
  if (entity) {
    const best = findBest(entity, 'macro_f1')
    if (best) cards.push({ label: 'Meilleur Entity (Raw)', model: best.model, value: pct(best.value), icon: Target, color: 'violet' })
  }

  // Best language
  const lang = data.language as SectionData | undefined
  if (lang) {
    const best = findBest(lang, 'accuracy')
    if (best) cards.push({ label: 'Meilleure Langue', model: best.model, value: pct(best.value), icon: Languages, color: 'amber' })
  }

  return cards
})

function findBest(section: SectionData, metric: string): { model: string; value: number } | null {
  let bestModel = ''
  let bestValue = -1
  for (const [model, metrics] of Object.entries(section)) {
    if (metrics[metric] !== undefined && metrics[metric] > bestValue) {
      bestValue = metrics[metric]
      bestModel = model
    }
  }
  return bestValue >= 0 ? { model: bestModel, value: bestValue } : null
}

// --- Per-class heatmap data ---
const perClassData = computed(() => {
  if (!activeResults.value) return null
  if (activeTab.value === 'combined' || activeTab.value === 'combined_fuzzy') return null

  const classGroups: Record<string, string[]> = {
    intent: ['trip', 'not_trip', 'unknown'],
    language: ['french', 'english', 'unknown'],
    entity: ['departure', 'destination'],
    entity_fuzzy: ['departure', 'destination'],
  }

  const classes = classGroups[activeTab.value]
  if (!classes) return null

  const models = Object.keys(activeResults.value)
  const rows: { cls: string; values: { model: string; p: number; r: number; f1: number }[] }[] = []

  for (const cls of classes) {
    const values = models.map(model => {
      const m = activeResults.value![model]
      return {
        model,
        p: m[`${cls}_precision`] || 0,
        r: m[`${cls}_recall`] || 0,
        f1: m[`${cls}_f1`] || 0,
      }
    })
    rows.push({ cls, values })
  }

  return { models, rows }
})

// --- Run config ---
const maxSamples = ref(2000)

async function handleRun() {
  await startEvaluation({
    eval_type: 'all',
    max_samples: maxSamples.value || undefined,
  })
}

const isEvalRunning = computed(
  () =>
    runLoading.value ||
    evalStatus.value?.status === 'running' ||
    evalStatus.value?.status === 'started',
)

// --- Export ---
function downloadReport() {
  if (!reportDetail.value) return
  const blob = new Blob([JSON.stringify(reportDetail.value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${selectedReportId.value || 'evaluation'}.json`
  a.click()
  URL.revokeObjectURL(url)
}

// --- Helpers ---
function pct(v: number | undefined): string {
  if (v === undefined || v === null) return '—'
  return (v * 100).toFixed(1) + '%'
}

function getMetricKeys(results: SectionData): string[] {
  const firstModel = Object.values(results)[0]
  if (!firstModel) return []
  // Filter out support and non-numeric keys
  return Object.keys(firstModel).filter(k => typeof firstModel[k] === 'number' && !k.endsWith('_support'))
}

function getMainMetrics(results: SectionData): string[] {
  return getMetricKeys(results).filter(k => ['accuracy', 'macro_f1', 'intent_accuracy', 'entity_accuracy', 'latency_ms'].includes(k))
}

function metricLabel(key: string): string {
  const labels: Record<string, string> = {
    accuracy: 'Accuracy',
    macro_f1: 'Macro F1',
    trip_precision: 'TRIP Prec',
    trip_recall: 'TRIP Rec',
    trip_f1: 'TRIP F1',
    not_trip_precision: 'NOT_TRIP Prec',
    not_trip_recall: 'NOT_TRIP Rec',
    not_trip_f1: 'NOT_TRIP F1',
    french_precision: 'FR Prec',
    french_recall: 'FR Rec',
    french_f1: 'FR F1',
    english_precision: 'EN Prec',
    english_recall: 'EN Rec',
    english_f1: 'EN F1',
    unknown_precision: 'UNK Prec',
    unknown_recall: 'UNK Rec',
    unknown_f1: 'UNK F1',
    departure_precision: 'Départ Prec',
    departure_recall: 'Départ Rec',
    departure_f1: 'Départ F1',
    destination_precision: 'Dest Prec',
    destination_recall: 'Dest Rec',
    destination_f1: 'Dest F1',
    intent_accuracy: 'Intent Acc',
    entity_accuracy: 'Entity Acc',
    latency_ms: 'Latence (ms)',
  }
  return labels[key] || key
}

function isBest(metric: string, value: number, results: SectionData): boolean {
  const allValues = Object.values(results).map(r => r[metric]).filter(v => v !== undefined)
  if (metric === 'latency_ms') return value === Math.min(...allValues)
  return value === Math.max(...allValues)
}

function isWorst(metric: string, value: number, results: SectionData): boolean {
  const allValues = Object.values(results).map(r => r[metric]).filter(v => v !== undefined)
  if (allValues.length < 2) return false
  if (metric === 'latency_ms') return value === Math.max(...allValues)
  return value === Math.min(...allValues)
}

function cellColor(metric: string, value: number, results: SectionData): string {
  if (metric === 'latency_ms') return 'text-gray-600'
  if (isBest(metric, value, results)) return 'font-bold text-green-700 bg-green-50'
  if (isWorst(metric, value, results)) return 'text-red-600 bg-red-50'
  return 'text-gray-600'
}

function heatColor(v: number): string {
  if (v >= 0.9) return 'bg-green-600 text-white'
  if (v >= 0.7) return 'bg-green-400 text-white'
  if (v >= 0.5) return 'bg-yellow-400 text-gray-900'
  if (v >= 0.3) return 'bg-orange-400 text-white'
  return 'bg-red-500 text-white'
}

function classLabel(cls: string): string {
  const labels: Record<string, string> = {
    trip: 'TRIP', not_trip: 'NOT_TRIP', unknown: 'UNKNOWN',
    french: 'Français', english: 'Anglais',
    departure: 'Départ', destination: 'Destination',
  }
  return labels[cls] || cls
}

// --- Init ---
onMounted(async () => {
  await loadReports()
  if (reports.value.length > 0) {
    selectedReportId.value = reports.value[0].id
  }
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

    <!-- ===================== Summary Cards ===================== -->
    <div v-if="summaryCards.length > 0" class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <div
        v-for="card in summaryCards"
        :key="card.label"
        class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm"
      >
        <div class="flex items-center gap-3">
          <div
            class="flex h-10 w-10 items-center justify-center rounded-lg"
            :class="{
              'bg-blue-50': card.color === 'blue',
              'bg-green-50': card.color === 'green',
              'bg-violet-50': card.color === 'violet',
              'bg-amber-50': card.color === 'amber',
            }"
          >
            <component
              :is="card.icon"
              class="h-5 w-5"
              :class="{
                'text-blue-600': card.color === 'blue',
                'text-green-600': card.color === 'green',
                'text-violet-600': card.color === 'violet',
                'text-amber-600': card.color === 'amber',
              }"
            />
          </div>
          <div>
            <p class="text-xs font-medium text-gray-500">{{ card.label }}</p>
            <p class="text-lg font-bold text-gray-900">{{ card.value }}</p>
            <p class="text-xs text-gray-500">{{ card.model }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- ===================== Lancer / Progress ===================== -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center justify-between border-b border-gray-200 px-6 py-4">
        <h2 class="text-base font-semibold text-gray-900">Lancer une évaluation</h2>
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
      <div class="px-6 py-4">
        <!-- Config -->
        <div v-if="!isEvalRunning" class="mb-4 flex flex-wrap items-center gap-4">
          <div class="flex items-center gap-1.5">
            <label class="text-xs font-medium text-gray-500">Échantillon :</label>
            <select
              v-model.number="maxSamples"
              class="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700"
            >
              <option :value="500">500 (rapide ~30min)</option>
              <option :value="1000">1 000 (~1h)</option>
              <option :value="2000">2 000 (~2h)</option>
              <option :value="5000">5 000 (~5h)</option>
              <option :value="0">Tout (15k, ~24h)</option>
            </select>
          </div>
        </div>

        <ErrorAlert v-if="runError" :message="runError" class="mb-4" />
        <EvalProgress v-if="isEvalRunning" :status="evalStatus" />
      </div>
    </div>

    <!-- ===================== Résultats ===================== -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center justify-between border-b border-gray-200 px-6 py-4">
        <h2 class="text-base font-semibold text-gray-900">Résultats</h2>

        <div class="flex items-center gap-3">
          <!-- Export -->
          <button
            v-if="reportDetail"
            class="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-2 text-xs font-medium text-gray-700 shadow-sm transition hover:bg-gray-50"
            @click="downloadReport"
          >
            <Download class="h-3.5 w-3.5" />
            Export JSON
          </button>

          <!-- Report selector -->
          <div v-if="reports.length > 0" class="relative">
            <select
              v-model="selectedReportId"
              class="appearance-none rounded-lg border border-gray-300 bg-white py-2 pl-3 pr-9 text-xs font-medium text-gray-700 shadow-sm transition hover:border-gray-400 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            >
              <option v-for="report in reports" :key="report.id" :value="report.id">
                {{ report.date }}
              </option>
            </select>
            <ChevronDown class="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" />
          </div>
        </div>
      </div>

      <div class="px-6 py-5">
        <!-- Loading -->
        <div v-if="reportsLoading || detailLoading" class="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </div>

        <ErrorAlert v-else-if="detailError" :message="detailError" />

        <!-- No reports -->
        <div v-else-if="reports.length === 0" class="rounded-lg bg-gray-50 py-12 text-center text-sm text-gray-400">
          Aucune évaluation disponible. Lancez une évaluation pour voir les résultats.
        </div>

        <!-- Results -->
        <div v-else-if="reportDetail">
          <!-- Tabs -->
          <div class="mb-5 flex flex-wrap gap-1 rounded-lg bg-gray-100 p-1">
            <button
              v-for="tab in tabs"
              :key="tab.key"
              class="rounded-md px-4 py-2 text-xs font-medium transition"
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

          <div v-if="activeResults && Object.keys(activeResults).length > 0" class="space-y-8">
            <!-- Main metrics table -->
            <div>
              <h3 class="mb-3 text-sm font-semibold text-gray-900">Métriques principales</h3>
              <div class="overflow-x-auto rounded-lg border border-gray-200">
                <table class="w-full text-sm">
                  <thead>
                    <tr class="bg-gray-50">
                      <th class="px-4 py-3 text-left font-semibold text-gray-900">Modèle</th>
                      <th
                        v-for="metric in getMainMetrics(activeResults)"
                        :key="metric"
                        class="px-3 py-3 text-right font-semibold text-gray-900"
                      >
                        {{ metricLabel(metric) }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(results, model) in activeResults"
                      :key="String(model)"
                      class="border-t border-gray-100 transition hover:bg-gray-50"
                    >
                      <td class="px-4 py-3 font-medium text-gray-900">{{ model }}</td>
                      <td
                        v-for="metric in getMainMetrics(activeResults)"
                        :key="metric"
                        class="px-3 py-3 text-right tabular-nums rounded"
                        :class="cellColor(metric, results[metric], activeResults)"
                      >
                        {{ metric.includes('latency') ? results[metric]?.toFixed(1) + ' ms' : pct(results[metric]) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Per-class heatmap -->
            <div v-if="perClassData">
              <h3 class="mb-3 text-sm font-semibold text-gray-900">Détail par classe (F1 Score)</h3>
              <div class="overflow-x-auto rounded-lg border border-gray-200">
                <table class="w-full text-sm">
                  <thead>
                    <tr class="bg-gray-50">
                      <th class="px-4 py-3 text-left font-semibold text-gray-900">Classe</th>
                      <th class="px-3 py-3 text-center font-semibold text-gray-900">Métrique</th>
                      <th
                        v-for="model in perClassData.models"
                        :key="model"
                        class="px-3 py-3 text-center font-semibold text-gray-900"
                      >
                        {{ model }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <template v-for="row in perClassData.rows" :key="row.cls">
                      <tr class="border-t border-gray-100">
                        <td rowspan="3" class="px-4 py-2 font-medium text-gray-900 align-top border-r border-gray-100">
                          {{ classLabel(row.cls) }}
                        </td>
                        <td class="px-3 py-1.5 text-center text-xs text-gray-500">Precision</td>
                        <td
                          v-for="v in row.values"
                          :key="v.model + '-p'"
                          class="px-3 py-1.5 text-center tabular-nums text-xs rounded-sm"
                          :class="heatColor(v.p)"
                        >
                          {{ pct(v.p) }}
                        </td>
                      </tr>
                      <tr>
                        <td class="px-3 py-1.5 text-center text-xs text-gray-500">Recall</td>
                        <td
                          v-for="v in row.values"
                          :key="v.model + '-r'"
                          class="px-3 py-1.5 text-center tabular-nums text-xs rounded-sm"
                          :class="heatColor(v.r)"
                        >
                          {{ pct(v.r) }}
                        </td>
                      </tr>
                      <tr class="border-b border-gray-200">
                        <td class="px-3 py-1.5 text-center text-xs font-medium text-gray-700">F1</td>
                        <td
                          v-for="v in row.values"
                          :key="v.model + '-f1'"
                          class="px-3 py-1.5 text-center tabular-nums text-xs font-medium rounded-sm"
                          :class="heatColor(v.f1)"
                        >
                          {{ pct(v.f1) }}
                        </td>
                      </tr>
                    </template>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Latency comparison -->
            <div v-if="!activeTab.startsWith('combined')">
              <h3 class="mb-3 text-sm font-semibold text-gray-900">Latence par modèle</h3>
              <div class="space-y-2">
                <div
                  v-for="(results, model) in activeResults"
                  :key="'lat-' + String(model)"
                  class="flex items-center gap-3"
                >
                  <span class="w-32 truncate text-sm font-medium text-gray-700">{{ model }}</span>
                  <div class="flex-1">
                    <div class="h-6 rounded-full bg-gray-100">
                      <div
                        class="flex h-6 items-center rounded-full bg-blue-500 px-2 text-xs font-medium text-white transition-all"
                        :style="{
                          width: Math.max(
                            5,
                            Math.min(100, (results.latency_ms / Math.max(...Object.values(activeResults!).map(r => r.latency_ms || 1))) * 100)
                          ) + '%'
                        }"
                      >
                        {{ results.latency_ms?.toFixed(1) }} ms
                      </div>
                    </div>
                  </div>
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

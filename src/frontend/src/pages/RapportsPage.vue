<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { FileText, Trophy, ChevronDown, Download, TrendingUp, Clock, Cpu } from 'lucide-vue-next'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import { useEvaluation } from '../composables/useEvaluation'

const {
  reports,
  reportsLoading,
  reportsError,
  reportDetail,
  detailLoading,
  detailError,
  loadReports,
  loadReport,
} = useEvaluation()

const selectedReportId = ref<string | null>(null)

watch(selectedReportId, async (id) => {
  if (id) await loadReport(id)
})

type MetricRow = Record<string, number>
type SectionData = Record<string, MetricRow>

// --- Best model per category ---
interface BestModel {
  category: string
  model: string
  metric: string
  value: number
  latency: number
}

const bestModels = computed((): BestModel[] => {
  if (!reportDetail.value) return []
  const data = reportDetail.value as Record<string, unknown>
  const results: BestModel[] = []

  const sections: { key: string; label: string; metric: string }[] = [
    { key: 'language', label: 'Détection de langue', metric: 'accuracy' },
    { key: 'intent', label: 'Classification intent', metric: 'accuracy' },
    { key: 'entity', label: 'Extraction entités (raw)', metric: 'macro_f1' },
    { key: 'entity_fuzzy', label: 'Extraction entités (fuzzy)', metric: 'macro_f1' },
  ]

  for (const section of sections) {
    const sectionData = data[section.key] as SectionData | undefined
    if (!sectionData) continue
    let bestModel = ''
    let bestValue = -1
    let bestLatency = 0
    for (const [model, metrics] of Object.entries(sectionData)) {
      if (metrics[section.metric] > bestValue) {
        bestValue = metrics[section.metric]
        bestModel = model
        bestLatency = metrics.latency_ms || 0
      }
    }
    if (bestModel) {
      results.push({
        category: section.label,
        model: bestModel,
        metric: section.metric,
        value: bestValue,
        latency: bestLatency,
      })
    }
  }
  return results
})

// --- Full comparison table per section ---
const sections = [
  { key: 'language', label: 'Langue' },
  { key: 'intent', label: 'Intent' },
  { key: 'entity', label: 'Entités' },
  { key: 'entity_fuzzy', label: 'Entités + Fuzzy' },
]

function getSectionData(key: string): SectionData | null {
  if (!reportDetail.value) return null
  const data = reportDetail.value as Record<string, unknown>
  return (data[key] as SectionData) || null
}

// --- Combined best pipeline ---
const bestPipeline = computed(() => {
  if (!reportDetail.value) return null
  const data = reportDetail.value as Record<string, unknown>
  const combined = data.combined_fuzzy as Array<Record<string, unknown>> | undefined
  if (!combined || !combined.length) return null

  let best = combined[0]
  for (const entry of combined) {
    const score = (entry.intent_accuracy as number) + (entry.entity_accuracy as number)
    const bestScore = (best.intent_accuracy as number) + (best.entity_accuracy as number)
    if (score > bestScore) best = entry
  }
  return best
})

// --- Helpers ---
function pct(v: number): string {
  return (v * 100).toFixed(1) + '%'
}

function cellBg(v: number): string {
  if (v >= 0.9) return 'bg-green-100 text-green-800'
  if (v >= 0.7) return 'bg-green-50 text-green-700'
  if (v >= 0.5) return 'bg-yellow-50 text-yellow-700'
  if (v >= 0.3) return 'bg-orange-50 text-orange-700'
  return 'bg-red-50 text-red-700'
}

function downloadReport() {
  if (!reportDetail.value) return
  const blob = new Blob([JSON.stringify(reportDetail.value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${selectedReportId.value || 'rapport'}.json`
  a.click()
  URL.revokeObjectURL(url)
}

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
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-50">
          <FileText class="h-6 w-6 text-amber-600" />
        </div>
        <div>
          <h1 class="text-2xl font-bold tracking-tight text-gray-900">Rapports</h1>
          <p class="text-gray-500">Synthèse des benchmarks et comparaison des modèles</p>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <button
          v-if="reportDetail"
          class="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-2 text-xs font-medium text-gray-700 shadow-sm transition hover:bg-gray-50"
          @click="downloadReport"
        >
          <Download class="h-3.5 w-3.5" />
          Export JSON
        </button>
        <div v-if="reports.length > 0" class="relative">
          <select
            v-model="selectedReportId"
            class="appearance-none rounded-lg border border-gray-300 bg-white py-2 pl-3 pr-9 text-xs font-medium text-gray-700 shadow-sm"
          >
            <option v-for="report in reports" :key="report.id" :value="report.id">
              {{ report.date }}
            </option>
          </select>
          <ChevronDown class="pointer-events-none absolute right-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" />
        </div>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="reportsLoading || detailLoading" class="flex items-center justify-center py-12">
      <LoadingSpinner size="lg" />
    </div>

    <ErrorAlert v-else-if="reportsError || detailError" :message="(reportsError || detailError)!" />

    <!-- No reports -->
    <div v-else-if="reports.length === 0" class="rounded-xl border border-gray-200 bg-white p-12 text-center shadow-sm">
      <FileText class="mx-auto h-12 w-12 text-gray-300" />
      <p class="mt-4 text-sm text-gray-500">Aucun rapport disponible. Lancez une évaluation depuis la page Évaluation.</p>
    </div>

    <template v-else-if="reportDetail">
      <!-- ===================== Best Models ===================== -->
      <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
        <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-4">
          <Trophy class="h-5 w-5 text-amber-500" />
          <h2 class="text-base font-semibold text-gray-900">Meilleur modèle par catégorie</h2>
        </div>
        <div class="grid grid-cols-1 divide-y divide-gray-100 sm:grid-cols-2 sm:divide-x sm:divide-y-0 lg:grid-cols-4">
          <div v-for="best in bestModels" :key="best.category" class="px-6 py-5">
            <p class="text-xs font-medium uppercase tracking-wide text-gray-400">{{ best.category }}</p>
            <p class="mt-1 text-2xl font-bold text-gray-900">{{ pct(best.value) }}</p>
            <p class="mt-0.5 text-sm text-gray-600">
              <span class="rounded bg-blue-50 px-1.5 py-0.5 text-xs font-medium text-blue-700">{{ best.model }}</span>
            </p>
            <p class="mt-1 text-xs text-gray-400">{{ best.latency.toFixed(1) }} ms / requête</p>
          </div>
        </div>
      </div>

      <!-- ===================== Best Pipeline ===================== -->
      <div v-if="bestPipeline" class="rounded-xl border border-gray-200 bg-gradient-to-r from-blue-50 to-violet-50 shadow-sm">
        <div class="px-6 py-5">
          <div class="flex items-center gap-2">
            <Cpu class="h-5 w-5 text-blue-600" />
            <h2 class="text-base font-semibold text-gray-900">Meilleure pipeline combinée (avec fuzzy)</h2>
          </div>
          <div class="mt-3 flex flex-wrap items-center gap-6">
            <div>
              <p class="text-xs text-gray-500">Intent</p>
              <p class="text-sm font-bold text-gray-900">{{ bestPipeline.intent }}</p>
              <p class="text-lg font-bold text-blue-700">{{ pct(bestPipeline.intent_accuracy as number) }}</p>
            </div>
            <div class="text-2xl text-gray-300">+</div>
            <div>
              <p class="text-xs text-gray-500">Entity</p>
              <p class="text-sm font-bold text-gray-900">{{ bestPipeline.entity }}</p>
              <p class="text-lg font-bold text-violet-700">{{ pct(bestPipeline.entity_accuracy as number) }}</p>
            </div>
            <div class="text-2xl text-gray-300">=</div>
            <div>
              <p class="text-xs text-gray-500">Score combiné</p>
              <p class="text-2xl font-bold text-gray-900">
                {{ pct(((bestPipeline.intent_accuracy as number) + (bestPipeline.entity_accuracy as number)) / 2) }}
              </p>
            </div>
            <div class="ml-auto flex items-center gap-1.5 text-sm text-gray-500">
              <Clock class="h-4 w-4" />
              {{ (bestPipeline.latency_ms as number).toFixed(1) }} ms
            </div>
          </div>
        </div>
      </div>

      <!-- ===================== Detailed Tables ===================== -->
      <div v-for="section in sections" :key="section.key" class="rounded-xl border border-gray-200 bg-white shadow-sm">
        <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-4">
          <TrendingUp class="h-5 w-5 text-gray-400" />
          <h2 class="text-base font-semibold text-gray-900">{{ section.label }}</h2>
        </div>

        <div v-if="getSectionData(section.key)" class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="bg-gray-50">
                <th class="px-4 py-3 text-left font-semibold text-gray-900">Modèle</th>
                <th class="px-3 py-3 text-right font-semibold text-gray-900">Accuracy</th>
                <th class="px-3 py-3 text-right font-semibold text-gray-900">Macro F1</th>
                <th
                  v-for="cls in getClasses(section.key)"
                  :key="cls"
                  class="px-3 py-3 text-right font-semibold text-gray-900"
                >
                  {{ classLabel(cls) }} F1
                </th>
                <th class="px-3 py-3 text-right font-semibold text-gray-900">Latence</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(metrics, model) in getSectionData(section.key)!"
                :key="String(model)"
                class="border-t border-gray-100 hover:bg-gray-50"
              >
                <td class="px-4 py-3 font-medium text-gray-900">{{ model }}</td>
                <td class="px-3 py-3 text-right tabular-nums rounded" :class="cellBg(metrics.accuracy)">
                  {{ pct(metrics.accuracy) }}
                </td>
                <td class="px-3 py-3 text-right tabular-nums rounded" :class="cellBg(metrics.macro_f1)">
                  {{ pct(metrics.macro_f1) }}
                </td>
                <td
                  v-for="cls in getClasses(section.key)"
                  :key="cls"
                  class="px-3 py-3 text-right tabular-nums rounded"
                  :class="cellBg(metrics[`${cls}_f1`] || 0)"
                >
                  {{ pct(metrics[`${cls}_f1`] || 0) }}
                </td>
                <td class="px-3 py-3 text-right tabular-nums text-gray-500">
                  {{ metrics.latency_ms?.toFixed(1) }} ms
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ===================== Analyse ===================== -->
      <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
        <div class="border-b border-gray-200 px-6 py-4">
          <h2 class="text-base font-semibold text-gray-900">Analyse</h2>
        </div>
        <div class="px-6 py-5 space-y-4 text-sm text-gray-700 leading-relaxed">
          <div v-if="bestModels.length > 0">
            <h3 class="font-semibold text-gray-900 mb-2">Points clés</h3>
            <ul class="list-disc pl-5 space-y-1.5">
              <li v-for="best in bestModels" :key="best.category">
                <strong>{{ best.category }}</strong> : {{ best.model }} atteint {{ pct(best.value) }}
                ({{ best.metric }}) avec {{ best.latency.toFixed(1) }} ms de latence.
              </li>
              <li>
                Le <strong>fuzzy matching</strong> améliore significativement les résultats d'extraction
                d'entités pour tous les modèles en normalisant les noms de gares vers la base SNCF officielle.
              </li>
              <li>
                <strong>SpaCy</strong> montre les limites d'un modèle NER générique non fine-tuné pour cette tâche :
                il détecte les entités LOC/GPE mais ne distingue pas départ, destination et étapes.
              </li>
              <li>
                <strong>Flan-T5 intent</strong> utilise le modèle base (non fine-tuné pour l'intent),
                ce qui explique ses performances faibles en classification. Son modèle fine-tuné
                est spécialisé en extraction d'entités uniquement.
              </li>
            </ul>
          </div>

          <div>
            <h3 class="font-semibold text-gray-900 mb-2">Progression méthodologique</h3>
            <p>
              Les résultats montrent une progression claire des approches :
              <strong>Regex</strong> (baseline par règles) →
              <strong>SpaCy</strong> (NER générique) →
              <strong>CamemBERT</strong> (fine-tuné, classification de tokens) →
              <strong>Flan-T5</strong> (fine-tuné, génération seq2seq).
              Chaque étape apporte une amélioration mesurable, justifiant l'investissement
              en complexité et en ressources de calcul.
            </p>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<script lang="ts">
function getClasses(sectionKey: string): string[] {
  const map: Record<string, string[]> = {
    language: ['french', 'english', 'unknown'],
    intent: ['trip', 'not_trip', 'unknown'],
    entity: ['departure', 'destination'],
    entity_fuzzy: ['departure', 'destination'],
  }
  return map[sectionKey] || []
}

function classLabel(cls: string): string {
  const labels: Record<string, string> = {
    french: 'FR', english: 'EN', unknown: 'UNK',
    trip: 'TRIP', not_trip: 'NOT_TRIP',
    departure: 'Départ', destination: 'Dest',
  }
  return labels[cls] || cls
}
</script>

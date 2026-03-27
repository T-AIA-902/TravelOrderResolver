<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { FileText, Trophy, Activity, Loader2, AlertCircle } from 'lucide-vue-next'
import { useEvaluation } from '../composables/useEvaluation'

const {
  reports,
  reportsLoading,
  reportsError,
  reportDetail,
  detailLoading,
  detailError,
  loadReports,
  loadLatestReport,
} = useEvaluation()

const benchmarkRows = [
  {
    step: 'Pré-traitement',
    model: 'STT Filter + Normalizer',
    accuracy: '\u2014',
    latency: '<1ms',
  },
  {
    step: 'Détection langue',
    model: 'Langdetect',
    accuracy: '77.9%',
    latency: '6.1ms',
  },
  {
    step: 'Classification intent',
    model: 'SpaCy',
    accuracy: '79.1%',
    latency: '0.8ms',
  },
  {
    step: 'Extraction entités',
    model: 'Regex + Fuzzy',
    accuracy: '57.1%',
    latency: '9.8ms',
  },
]

/** Flatten the latest report detail into displayable metric rows. */
const liveMetricRows = computed(() => {
  const detail = reportDetail.value
  if (!detail) return []

  const rows: { category: string; model: string; accuracy: string; latency: string }[] = []

  const sections: { key: keyof typeof detail; label: string }[] = [
    { key: 'language', label: 'Détection langue' },
    { key: 'intent', label: 'Classification intent' },
    { key: 'entity', label: 'Extraction entités' },
    { key: 'entity_fuzzy', label: 'Extraction entités (fuzzy)' },
  ]

  for (const section of sections) {
    const data = detail[section.key] as Record<string, { accuracy: number; avg_latency_ms: number }> | undefined
    if (!data) continue
    for (const [modelName, result] of Object.entries(data)) {
      rows.push({
        category: section.label,
        model: modelName,
        accuracy: `${(result.accuracy * 100).toFixed(1)}%`,
        latency: `${result.avg_latency_ms.toFixed(1)}ms`,
      })
    }
  }

  return rows
})

const hasLiveData = computed(() => reports.value.length > 0)
const isLoading = computed(() => reportsLoading.value || detailLoading.value)
const error = computed(() => reportsError.value || detailError.value)

onMounted(async () => {
  await loadReports()
  if (reports.value.length > 0) {
    await loadLatestReport()
  }
})
</script>

<template>
  <div class="space-y-6 p-8">
    <!-- Header -->
    <div class="flex items-center gap-4">
      <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-50">
        <FileText class="h-6 w-6 text-amber-600" />
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-gray-900">Rapports</h1>
        <p class="text-gray-500">Benchmark et rapports d'évaluation</p>
      </div>
    </div>

    <!-- Loading state -->
    <div v-if="isLoading" class="flex items-center gap-3 rounded-xl border border-gray-200 bg-white px-6 py-5 shadow-sm">
      <Loader2 class="h-5 w-5 animate-spin text-amber-500" />
      <span class="text-sm text-gray-600">Chargement des rapports...</span>
    </div>

    <!-- Error state -->
    <div v-if="error" class="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-6 py-5">
      <AlertCircle class="h-5 w-5 text-red-500" />
      <span class="text-sm text-red-700">{{ error }}</span>
    </div>

    <!-- Live evaluation results -->
    <div v-if="hasLiveData && !isLoading" class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Activity class="h-5 w-5 text-emerald-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Dernière évaluation</h2>
          <p class="text-sm text-gray-500">
            {{ reportDetail?.config?.eval_type ?? reports[0]?.eval_type }} &mdash;
            {{ reportDetail?.config?.samples ?? reports[0]?.samples }} exemples &mdash;
            {{ reports[0]?.date }}
          </p>
        </div>
      </div>

      <!-- Report list -->
      <div v-if="reports.length > 1" class="border-b border-gray-200 px-6 py-3">
        <p class="text-xs font-medium uppercase tracking-wide text-gray-400">
          {{ reports.length }} rapports disponibles
        </p>
      </div>

      <!-- Metrics table -->
      <div v-if="liveMetricRows.length" class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Catégorie</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Modèle</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Accuracy</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Latence</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in liveMetricRows"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4 font-medium text-gray-900">{{ row.category }}</td>
              <td class="px-6 py-4">
                <span class="rounded-md bg-emerald-50 px-2.5 py-1 text-sm font-medium text-emerald-700">
                  {{ row.model }}
                </span>
              </td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.accuracy }}</td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.latency }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div v-else class="px-6 py-5 text-sm text-gray-500">
        Le rapport ne contient pas de métriques détaillées.
      </div>
    </div>

    <!-- Static benchmarks -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Trophy class="h-5 w-5 text-amber-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Benchmarks de référence</h2>
          <p class="text-sm text-gray-500">Meilleurs résultats par étape du pipeline</p>
        </div>
      </div>

      <div v-if="!hasLiveData && !isLoading" class="border-b border-gray-100 bg-amber-50/50 px-6 py-3">
        <p class="text-sm text-amber-700">
          Aucune évaluation live n'a été lancée. Les données ci-dessous sont des benchmarks statiques.
        </p>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50/80">
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Étape</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">
                Meilleur modèle
              </th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Accuracy</th>
              <th class="px-6 py-3.5 text-left text-sm font-medium text-gray-600">Latence</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in benchmarkRows"
              :key="i"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-6 py-4 font-medium text-gray-900">{{ row.step }}</td>
              <td class="px-6 py-4">
                <span class="rounded-md bg-gray-100 px-2.5 py-1 text-sm font-medium text-gray-700">
                  {{ row.model }}
                </span>
              </td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.accuracy }}</td>
              <td class="px-6 py-4 font-mono text-gray-600">{{ row.latency }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

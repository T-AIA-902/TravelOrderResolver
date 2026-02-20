<script setup lang="ts">
import { FileText, Trophy } from 'lucide-vue-next'

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

    <!-- Benchmark -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex items-center gap-3 border-b border-gray-200 px-6 py-5">
        <Trophy class="h-5 w-5 text-amber-500" />
        <div>
          <h2 class="text-lg font-semibold text-gray-900">Benchmark</h2>
          <p class="text-sm text-gray-500">Meilleurs résultats par étape du pipeline</p>
        </div>
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

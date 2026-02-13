<script setup lang="ts">
import type { ModelEvalResult } from '../../api/types'

defineProps<{
  modelResults: Record<string, ModelEvalResult>
}>()

function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + '%'
}

function formatLatency(ms: number): string {
  return ms.toFixed(1) + ' ms'
}
</script>

<template>
  <div class="space-y-6">
    <div v-for="(result, model) in modelResults" :key="model">
      <!-- Model header -->
      <div class="mb-3 flex items-center justify-between">
        <h4 class="font-semibold text-gray-900">{{ model }}</h4>
        <div class="flex gap-5 text-sm text-gray-600">
          <span>
            Accuracy : <strong class="text-gray-900">{{ formatPercent(result.accuracy) }}</strong>
          </span>
          <span>
            Latence moy. :
            <strong class="text-gray-900">{{ formatLatency(result.avg_latency_ms) }}</strong>
          </span>
        </div>
      </div>

      <!-- Per-class table -->
      <div class="overflow-x-auto rounded-lg border border-gray-200">
        <table class="w-full">
          <thead>
            <tr class="border-b border-gray-200 bg-gray-50">
              <th class="px-5 py-3 text-left text-sm font-medium text-gray-600">Classe</th>
              <th class="px-5 py-3 text-left text-sm font-medium text-gray-600">Précision</th>
              <th class="px-5 py-3 text-left text-sm font-medium text-gray-600">Rappel</th>
              <th class="px-5 py-3 text-left text-sm font-medium text-gray-600">F1</th>
              <th class="px-5 py-3 text-left text-sm font-medium text-gray-600">Support</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(metrics, className) in result.per_class"
              :key="className"
              class="border-b border-gray-100 transition-colors last:border-0 hover:bg-gray-50/50"
            >
              <td class="px-5 py-3.5 font-medium text-gray-800">{{ className }}</td>
              <td class="px-5 py-3.5 font-mono text-gray-600">
                {{ formatPercent(metrics.precision) }}
              </td>
              <td class="px-5 py-3.5 font-mono text-gray-600">
                {{ formatPercent(metrics.recall) }}
              </td>
              <td class="px-5 py-3.5 font-mono text-gray-600">{{ formatPercent(metrics.f1) }}</td>
              <td class="px-5 py-3.5 font-mono text-gray-600">{{ metrics.support }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMonitoring } from '../composables/useMonitoring'
import ResourceGauge from '../components/monitoring/ResourceGauge.vue'
import CarbonCard from '../components/monitoring/CarbonCard.vue'
import LatencyChart from '../components/monitoring/LatencyChart.vue'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import { RefreshCw } from 'lucide-vue-next'

const { metrics, resources, carbon, isLoading, error, refresh, totalRequests, avgLatency } =
  useMonitoring()
</script>

<template>
  <div class="space-y-6 p-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold text-gray-900">Monitoring</h1>
        <p class="mt-1 text-sm text-gray-500">Ressources système, latence et empreinte carbone</p>
      </div>
      <button
        class="flex items-center gap-2 rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-600 transition hover:bg-gray-50"
        :disabled="isLoading"
        @click="refresh"
      >
        <RefreshCw class="h-4 w-4" :class="{ 'animate-spin': isLoading }" />
        Rafraîchir
      </button>
    </div>

    <ErrorAlert v-if="error" :message="error" />

    <!-- Stats cards -->
    <div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <div class="rounded-xl border border-gray-200 bg-white p-5">
        <p class="text-sm text-gray-500">Requêtes totales</p>
        <p class="mt-1 text-2xl font-bold text-gray-900">{{ totalRequests }}</p>
      </div>
      <div class="rounded-xl border border-gray-200 bg-white p-5">
        <p class="text-sm text-gray-500">Latence moyenne</p>
        <p class="mt-1 text-2xl font-bold text-gray-900">{{ avgLatency.toFixed(0) }} ms</p>
      </div>
      <div class="rounded-xl border border-gray-200 bg-white p-5">
        <p class="text-sm text-gray-500">Statut</p>
        <p class="mt-1 text-2xl font-bold" :class="error ? 'text-red-500' : 'text-emerald-500'">
          {{ error ? 'Hors ligne' : 'En ligne' }}
        </p>
      </div>
    </div>

    <!-- Resources -->
    <div class="rounded-xl border border-gray-200 bg-white p-6">
      <h2 class="mb-4 text-sm font-semibold text-gray-900">Ressources système</h2>
      <div v-if="resources" class="space-y-4">
        <ResourceGauge label="CPU" :value="resources.cpu_percent" :max="100" unit="%" />
        <ResourceGauge
          label="RAM"
          :value="resources.ram_used_mb"
          :max="resources.ram_total_mb"
          unit=" MB"
          color="bg-violet-500"
        />
        <ResourceGauge
          v-if="resources.gpu_used_mb != null && resources.gpu_total_mb != null"
          label="GPU"
          :value="resources.gpu_used_mb"
          :max="resources.gpu_total_mb"
          unit=" MB"
          color="bg-emerald-500"
        />
      </div>
      <LoadingSpinner v-else-if="isLoading" />
      <p v-else class="text-sm text-gray-400">En attente du backend...</p>
    </div>

    <!-- Charts row -->
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <LatencyChart :metrics="metrics" />
      <CarbonCard :carbon="carbon" />
    </div>

    <!-- Request history table -->
    <div class="rounded-xl border border-gray-200 bg-white p-6">
      <h2 class="mb-4 text-sm font-semibold text-gray-900">Historique des requêtes</h2>
      <div v-if="metrics.length > 0" class="overflow-x-auto">
        <table class="w-full text-left text-sm">
          <thead>
            <tr class="border-b border-gray-100 text-xs uppercase text-gray-500">
              <th class="pb-2 pr-4">Heure</th>
              <th class="pb-2 pr-4">Texte</th>
              <th class="pb-2 pr-4">Modèle</th>
              <th class="pb-2 pr-4">Latence</th>
              <th class="pb-2 pr-4">CPU</th>
              <th class="pb-2">RAM</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(m, i) in metrics.slice(0, 20)" :key="i" class="border-b border-gray-50">
              <td class="py-2 pr-4 text-gray-500">
                {{ new Date(m.timestamp).toLocaleTimeString('fr-FR') }}
              </td>
              <td class="max-w-48 truncate py-2 pr-4 text-gray-700">{{ m.input_text }}</td>
              <td class="py-2 pr-4">
                <span class="rounded bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">
                  {{ m.model_name }}
                </span>
              </td>
              <td class="py-2 pr-4 text-gray-700">{{ (m.duration_s * 1000).toFixed(0) }}ms</td>
              <td class="py-2 pr-4 text-gray-500">
                {{ m.cpu_percent_avg != null ? m.cpu_percent_avg.toFixed(0) + '%' : '-' }}
              </td>
              <td class="py-2 text-gray-500">
                {{ m.ram_peak_mb != null ? m.ram_peak_mb.toFixed(0) + 'MB' : '-' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="text-sm text-gray-400">Aucune requête enregistrée</p>
    </div>
  </div>
</template>

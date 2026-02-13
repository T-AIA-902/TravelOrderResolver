<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink } from 'vue-router'
import {
  MessageSquare,
  BarChart3,
  Map,
  ExternalLink,
  Activity,
  Train,
  Cpu,
  Monitor,
} from 'lucide-vue-next'
import StatusDot from '../components/common/StatusDot.vue'
import LoadingSpinner from '../components/common/LoadingSpinner.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import { apiGet } from '../api/client'
import type { HealthResponse } from '../api/types'

const health = ref<HealthResponse | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

onMounted(async () => {
  try {
    health.value = await apiGet<HealthResponse>('/api/health')
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Impossible de contacter le serveur'
  } finally {
    loading.value = false
  }
})

const isHealthy = () => health.value?.status === 'ok'

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

const quickActions = [
  {
    label: 'Tester le pipeline',
    to: '/chat',
    icon: MessageSquare,
    external: false,
  },
  {
    label: 'Lancer une évaluation',
    to: '/evaluation',
    icon: BarChart3,
    external: false,
  },
  {
    label: 'Explorer la carte',
    to: '/map',
    icon: Map,
    external: false,
  },
  {
    label: 'Ouvrir Grafana',
    to: 'http://localhost:3000',
    icon: ExternalLink,
    external: true,
  },
]
</script>

<template>
  <div class="space-y-8 p-6">
    <!-- Header -->
    <div>
      <h1 class="text-2xl font-bold text-gray-900">Dashboard</h1>
      <p class="text-gray-600">Vue d'ensemble du système</p>
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="flex items-center justify-center py-12">
      <LoadingSpinner size="lg" />
    </div>

    <!-- Error state -->
    <ErrorAlert v-if="error" :message="error" />

    <!-- Status cards -->
    <div v-if="!loading" class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <!-- API Status -->
      <div class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div class="flex items-center gap-3 mb-3">
          <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-50">
            <Activity class="h-5 w-5 text-blue-600" />
          </div>
          <span class="text-sm font-medium text-gray-500">API Status</span>
        </div>
        <div class="flex items-center gap-2">
          <StatusDot :healthy="!!health && isHealthy()" />
          <span class="text-lg font-semibold text-gray-900">
            {{ health && isHealthy() ? 'Connecté' : 'Déconnecté' }}
          </span>
        </div>
      </div>

      <!-- Gares -->
      <div class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div class="flex items-center gap-3 mb-3">
          <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50">
            <Train class="h-5 w-5 text-emerald-600" />
          </div>
          <span class="text-sm font-medium text-gray-500">Gares</span>
        </div>
        <span class="text-lg font-semibold text-gray-900">
          {{ health?.stations_count ?? '\u2014' }}
        </span>
      </div>

      <!-- Modeles -->
      <div class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div class="flex items-center gap-3 mb-3">
          <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-violet-50">
            <Cpu class="h-5 w-5 text-violet-600" />
          </div>
          <span class="text-sm font-medium text-gray-500">Modèles</span>
        </div>
        <span class="text-lg font-semibold text-gray-900">
          {{ health?.models_loaded ? health.models_loaded.length : '\u2014' }}
        </span>
      </div>

      <!-- Device -->
      <div class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div class="flex items-center gap-3 mb-3">
          <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-amber-50">
            <Monitor class="h-5 w-5 text-amber-600" />
          </div>
          <span class="text-sm font-medium text-gray-500">Device</span>
        </div>
        <span class="text-lg font-semibold text-gray-900">
          {{ health?.gpu_name ?? health?.device ?? '\u2014' }}
        </span>
      </div>
    </div>

    <!-- Benchmark section -->
    <div v-if="!loading" class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="border-b border-gray-200 px-5 py-4">
        <h2 class="text-lg font-semibold text-gray-900">Benchmark</h2>
        <p class="text-sm text-gray-500">Meilleurs résultats par étape du pipeline</p>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-gray-100 bg-gray-50">
              <th class="px-5 py-3 text-left font-medium text-gray-600">Étape</th>
              <th class="px-5 py-3 text-left font-medium text-gray-600">Meilleur modèle</th>
              <th class="px-5 py-3 text-left font-medium text-gray-600">Accuracy</th>
              <th class="px-5 py-3 text-left font-medium text-gray-600">Latence</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="(row, i) in benchmarkRows"
              :key="i"
              class="border-b border-gray-50 last:border-0"
            >
              <td class="px-5 py-3 font-medium text-gray-900">{{ row.step }}</td>
              <td class="px-5 py-3 text-gray-600">{{ row.model }}</td>
              <td class="px-5 py-3 text-gray-600">{{ row.accuracy }}</td>
              <td class="px-5 py-3 text-gray-600">{{ row.latency }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Actions rapides -->
    <div v-if="!loading">
      <h2 class="mb-4 text-lg font-semibold text-gray-900">Actions rapides</h2>
      <div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <template v-for="action in quickActions" :key="action.label">
          <!-- External link -->
          <a
            v-if="action.external"
            :href="action.to"
            target="_blank"
            rel="noopener noreferrer"
            class="flex items-center gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm transition hover:border-blue-300 hover:shadow-md"
          >
            <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
              <component :is="action.icon" class="h-5 w-5 text-gray-700" />
            </div>
            <span class="text-sm font-medium text-gray-900">{{ action.label }}</span>
          </a>
          <!-- Internal router link -->
          <RouterLink
            v-else
            :to="action.to"
            class="flex items-center gap-3 rounded-xl border border-gray-200 bg-white p-4 shadow-sm transition hover:border-blue-300 hover:shadow-md"
          >
            <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100">
              <component :is="action.icon" class="h-5 w-5 text-gray-700" />
            </div>
            <span class="text-sm font-medium text-gray-900">{{ action.label }}</span>
          </RouterLink>
        </template>
      </div>
    </div>

    <!-- A propos -->
    <div v-if="!loading" class="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <h2 class="mb-3 text-lg font-semibold text-gray-900">À propos</h2>
      <dl class="grid grid-cols-1 gap-2 text-sm sm:grid-cols-3">
        <div>
          <dt class="text-gray-500">Projet</dt>
          <dd class="font-medium text-gray-900">Travel Order Resolver</dd>
        </div>
        <div>
          <dt class="text-gray-500">Version</dt>
          <dd class="font-medium text-gray-900">{{ health?.version ?? '\u2014' }}</dd>
        </div>
        <div>
          <dt class="text-gray-500">Équipe</dt>
          <dd class="font-medium text-gray-900">T-AIA-902</dd>
        </div>
      </dl>
    </div>
  </div>
</template>

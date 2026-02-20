<script setup lang="ts">
import { RouterLink } from 'vue-router'
import { MessageSquare, BarChart3, Database, FileText, Activity, Train } from 'lucide-vue-next'

const quickActions = [
  {
    label: 'Tester le pipeline',
    description: 'Chat conversationnel avec analyse NLP',
    to: '/chat',
    icon: MessageSquare,
    color: 'bg-blue-50 text-blue-600',
  },
  {
    label: 'Évaluation',
    description: 'Lancer et consulter les évaluations',
    to: '/evaluation',
    icon: BarChart3,
    color: 'bg-violet-50 text-violet-600',
  },
  {
    label: 'Dataset',
    description: 'Répartition et métriques du dataset',
    to: '/dataset',
    icon: Database,
    color: 'bg-emerald-50 text-emerald-600',
  },
  {
    label: 'Rapports',
    description: 'Benchmark et résultats par étape',
    to: '/rapports',
    icon: FileText,
    color: 'bg-amber-50 text-amber-600',
  },
  {
    label: 'Monitoring',
    description: 'Ressources, latence et empreinte carbone',
    to: '/monitoring',
    icon: Activity,
    color: 'bg-rose-50 text-rose-600',
  },
]
</script>

<template>
  <div class="flex min-h-screen flex-col items-center justify-center px-6">
    <!-- Hero -->
    <div class="mb-12 text-center">
      <div class="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-900">
        <Train class="h-8 w-8 text-white" />
      </div>
      <h1 class="text-4xl font-bold tracking-tight text-gray-900">Travel Order Resolver</h1>
      <p class="mx-auto mt-3 max-w-lg text-lg text-gray-500">
        Pipeline NLP pour la résolution de commandes de voyage ferroviaire — détection de langue,
        classification d'intent et extraction d'entités.
      </p>
    </div>

    <!-- Actions -->
    <div class="grid w-full max-w-3xl grid-cols-1 gap-4 sm:grid-cols-2">
      <RouterLink
        v-for="action in quickActions"
        :key="action.label"
        :to="action.to"
        class="flex items-start gap-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition hover:border-blue-300 hover:shadow-md"
      >
        <div
          class="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg"
          :class="action.color"
        >
          <component :is="action.icon" class="h-5 w-5" />
        </div>
        <div>
          <span class="text-sm font-semibold text-gray-900">{{ action.label }}</span>
          <p class="mt-0.5 text-sm text-gray-500">{{ action.description }}</p>
        </div>
      </RouterLink>
    </div>

    <!-- Footer -->
    <p class="mt-12 text-xs text-gray-400">T-AIA-902 — Epitech</p>
  </div>
</template>

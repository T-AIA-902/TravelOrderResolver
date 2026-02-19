<script setup lang="ts">
import { Database, BarChart2 } from 'lucide-vue-next'
</script>

<template>
  <div class="space-y-6 p-8">
    <!-- Header -->
    <div class="flex items-center gap-4">
      <div class="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50">
        <Database class="h-6 w-6 text-emerald-600" />
      </div>
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-gray-900">Dataset</h1>
        <p class="text-gray-500">Répartition et métriques du dataset</p>
      </div>
    </div>

    <!-- Placeholder -->
    <div class="rounded-xl border border-gray-200 bg-white shadow-sm">
      <div class="flex flex-col items-center px-6 py-20">
        <div class="mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-gray-100">
          <BarChart2 class="h-7 w-7 text-gray-400" />
        </div>
        <p class="font-medium text-gray-500">Bientôt disponible</p>
        <p class="mt-2 max-w-sm text-center text-gray-400">
          Les courbes de répartition et métriques du dataset seront ajoutées ici.
        </p>
      </div>
    </div>
  </div>
</template>

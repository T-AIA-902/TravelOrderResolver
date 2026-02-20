<script setup lang="ts">
import { Leaf } from 'lucide-vue-next'
import type { CarbonSummary } from '../../api/types'

defineProps<{
  carbon: CarbonSummary | null
}>()
</script>

<template>
  <div class="rounded-xl border border-gray-200 bg-white p-6">
    <div class="mb-4 flex items-center gap-3">
      <div class="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-50">
        <Leaf class="h-5 w-5 text-emerald-600" />
      </div>
      <h3 class="text-sm font-semibold text-gray-900">Empreinte carbone</h3>
    </div>

    <div v-if="carbon" class="grid grid-cols-2 gap-4">
      <div>
        <p class="text-2xl font-bold text-gray-900">
          {{ (carbon.total_emissions_kg * 1000).toFixed(2) }}
        </p>
        <p class="text-xs text-gray-500">g CO2 total</p>
      </div>
      <div>
        <p class="text-2xl font-bold text-gray-900">
          {{ (carbon.total_energy_kwh * 1000).toFixed(1) }}
        </p>
        <p class="text-xs text-gray-500">Wh consommés</p>
      </div>
      <div>
        <p class="text-lg font-semibold text-gray-700">{{ carbon.total_requests }}</p>
        <p class="text-xs text-gray-500">Requêtes totales</p>
      </div>
      <div>
        <p class="text-lg font-semibold text-gray-700">{{ carbon.country }}</p>
        <p class="text-xs text-gray-500">Pays</p>
      </div>
    </div>

    <div v-else class="text-sm text-gray-400">Aucune donnée carbone disponible</div>
  </div>
</template>

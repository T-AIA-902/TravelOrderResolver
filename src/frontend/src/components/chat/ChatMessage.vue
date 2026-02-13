<script setup lang="ts">
import type { NlpResult, PathfindingResult } from '../../api/types'
import PipelineVisualizer from '../nlp/PipelineVisualizer.vue'
import { MapPin, AlertCircle } from 'lucide-vue-next'

defineProps<{
  role: 'user' | 'system'
  text: string
  nlp?: NlpResult
  pathfinding?: PathfindingResult
}>()
</script>

<template>
  <div class="flex w-full" :class="role === 'user' ? 'justify-end' : 'justify-start'">
    <div
      class="rounded-xl px-4 py-3"
      :class="
        role === 'user'
          ? 'max-w-[80%] bg-blue-50 text-gray-900'
          : 'max-w-[85%] border border-gray-200 bg-white text-gray-900'
      "
    >
      <!-- Message text -->
      <p class="whitespace-pre-wrap text-sm leading-relaxed">{{ text }}</p>

      <!-- NLP pipeline visualization (system only) -->
      <div v-if="role === 'system' && nlp" class="mt-3 border-t border-gray-100 pt-3">
        <PipelineVisualizer :nlp="nlp" />
      </div>

      <!-- Pathfinding result (system only) -->
      <div v-if="role === 'system' && pathfinding" class="mt-3 border-t border-gray-100 pt-3">
        <!-- Route found -->
        <div
          v-if="pathfinding.found"
          class="flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-800"
        >
          <MapPin class="h-4 w-4 shrink-0" />
          <span>
            Itinéraire trouvé : {{ pathfinding.total_stops }} arrêts,
            {{ pathfinding.transfers }} correspondance(s)
          </span>
        </div>

        <!-- Route not found -->
        <div
          v-else
          class="flex items-start gap-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700"
        >
          <AlertCircle class="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p class="font-medium">Aucun itinéraire trouvé</p>
            <p v-if="pathfinding.error" class="mt-1 text-red-600">
              {{ pathfinding.error }}
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

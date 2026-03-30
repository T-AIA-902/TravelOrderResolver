<script setup lang="ts">
import { ref } from 'vue'
import type { NlpResult, PathfindingResult } from '../../api/types'
import PipelineVisualizer from '../nlp/PipelineVisualizer.vue'
import { MapPin, AlertCircle, Route, Clock, Ruler, ArrowRightLeft } from 'lucide-vue-next'

const props = defineProps<{
  role: 'user' | 'system'
  text: string
  nlp?: NlpResult
  pathfinding?: PathfindingResult
}>()

const emit = defineEmits<{
  selectRoute: [index: number]
}>()

const selectedPareto = ref(0)
const expandedRoute = ref<number | null>(null)

function selectRoute(index: number) {
  selectedPareto.value = index
  expandedRoute.value = expandedRoute.value === index ? null : index
  emit('selectRoute', index)
}

// Determine what makes each route special
function routeTag(index: number): { label: string; color: string } | null {
  const routes = props.pathfinding?.pareto_routes
  if (!routes || routes.length < 2) return null

  const r = routes[index]
  const isMinTime = routes.every(o => r.cost.time_h <= o.cost.time_h)
  const isMinDist = routes.every(o => r.cost.distance_km <= o.cost.distance_km)
  const isMinTransfers = routes.every(o => r.cost.transfers <= o.cost.transfers)

  if (isMinTime && !isMinDist) return { label: 'Plus rapide', color: 'bg-blue-100 text-blue-800' }
  if (isMinDist && !isMinTime) return { label: 'Plus courte', color: 'bg-green-100 text-green-800' }
  if (isMinTransfers && !isMinTime && !isMinDist) return { label: 'Moins de corresp.', color: 'bg-amber-100 text-amber-800' }
  if (isMinTime && isMinDist) return { label: 'Optimale', color: 'bg-violet-100 text-violet-800' }
  return null
}
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
        <div v-if="pathfinding.found">
          <div class="flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-800">
            <MapPin class="h-4 w-4 shrink-0" />
            <span>
              Itinéraire trouvé
              <span v-if="pathfinding.algorithm" class="text-green-600">({{ pathfinding.algorithm.toUpperCase() }})</span>
              : {{ pathfinding.total_stops }} arrêts,
              {{ pathfinding.transfers }} correspondance(s)
            </span>
          </div>

          <!-- Pareto alternatives (MOA*) -->
          <div v-if="pathfinding.pareto_routes && pathfinding.pareto_routes.length > 1" class="mt-3 space-y-2">
            <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              <Route class="inline h-3.5 w-3.5" />
              {{ pathfinding.pareto_routes.length }} alternatives Pareto
            </p>
            <div class="space-y-1.5">
              <div
                v-for="(route, i) in pathfinding.pareto_routes"
                :key="i"
              >
                <button
                  class="flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left text-xs transition"
                  :class="selectedPareto === i
                    ? 'border-blue-400 bg-blue-50 ring-1 ring-blue-400'
                    : 'border-gray-200 bg-gray-50 hover:bg-gray-100'
                  "
                  @click="selectRoute(i)"
                >
                  <span class="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-bold"
                    :class="selectedPareto === i ? 'bg-blue-600 text-white' : 'bg-gray-300 text-white'"
                  >
                    {{ i + 1 }}
                  </span>
                  <div class="flex flex-1 flex-wrap items-center gap-2">
                    <span v-if="routeTag(i)" class="rounded-full px-2 py-0.5 text-[10px] font-semibold" :class="routeTag(i)!.color">
                      {{ routeTag(i)!.label }}
                    </span>
                    <span class="inline-flex items-center gap-1 text-gray-700">
                      <Clock class="h-3 w-3 text-gray-400" />
                      {{ route.cost.time_h }}h
                    </span>
                    <span class="inline-flex items-center gap-1 text-gray-700">
                      <Ruler class="h-3 w-3 text-gray-400" />
                      {{ route.cost.distance_km }} km
                    </span>
                    <span class="inline-flex items-center gap-1 text-gray-700">
                      <ArrowRightLeft class="h-3 w-3 text-gray-400" />
                      {{ route.cost.transfers }} corresp.
                    </span>
                  </div>
                  <span class="shrink-0 text-[10px] text-gray-400">
                    {{ route.path.length }} gares ▾
                  </span>
                </button>
                <!-- Expanded: station list -->
                <div
                  v-if="expandedRoute === i"
                  class="ml-9 mt-1 mb-1 rounded-lg border border-gray-100 bg-gray-50 px-3 py-2"
                >
                  <div class="flex flex-wrap items-center gap-1 text-[11px] text-gray-600">
                    <template v-for="(station, si) in route.path" :key="si">
                      <span class="font-medium" :class="si === 0 || si === route.path.length - 1 ? 'text-gray-900' : ''">
                        {{ station }}
                      </span>
                      <span v-if="si < route.path.length - 1" class="text-gray-300">→</span>
                    </template>
                  </div>
                </div>
              </div>
            </div>
          </div>
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

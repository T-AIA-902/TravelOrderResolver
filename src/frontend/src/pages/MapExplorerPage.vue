<script setup lang="ts">
import { ref } from 'vue'
import { searchStations } from '../api/pathfinding'
import type { Station } from '../api/types'
import RailwayMap from '../components/map/RailwayMap.vue'
import { Search } from 'lucide-vue-next'

const query = ref('')
const results = ref<Station[]>([])
const searching = ref(false)
const selectedStation = ref<Station | null>(null)

async function onSearch() {
  const q = query.value.trim()
  if (!q) return
  searching.value = true
  try {
    const data = await searchStations(q, 10)
    results.value = data.stations
  } catch {
    results.value = []
  } finally {
    searching.value = false
  }
}

function selectStation(station: Station) {
  selectedStation.value = station
}
</script>

<template>
  <div class="flex h-full flex-col">
    <!-- Search bar -->
    <div class="border-b border-gray-200 bg-white px-6 py-4">
      <div class="flex items-center gap-3">
        <div class="relative flex-1">
          <Search class="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
          <input
            v-model="query"
            type="text"
            placeholder="Rechercher une gare..."
            class="w-full rounded-lg border border-gray-200 bg-gray-50 py-2 pl-10 pr-4 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-400"
            @keydown.enter="onSearch"
          />
        </div>
        <button
          :disabled="searching || !query.trim()"
          class="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          @click="onSearch"
        >
          Rechercher
        </button>
      </div>

      <!-- Results list -->
      <div v-if="results.length > 0" class="mt-3 flex flex-wrap gap-2">
        <button
          v-for="station in results"
          :key="station.uic"
          class="rounded-full border px-3 py-1 text-xs transition-colors"
          :class="
            selectedStation?.uic === station.uic
              ? 'border-blue-400 bg-blue-50 text-blue-700'
              : 'border-gray-200 bg-white text-gray-700 hover:border-blue-300'
          "
          @click="selectStation(station)"
        >
          {{ station.name }}
        </button>
      </div>
    </div>

    <!-- Map -->
    <div class="flex-1">
      <RailwayMap :departure-station="selectedStation" height="100%" />
    </div>
  </div>
</template>

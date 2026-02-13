<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import StatusDot from '../common/StatusDot.vue'

const route = useRoute()

const healthData = ref<{
  status: string
  version?: string
  gpu_name?: string
  stations_count?: number
} | null>(null)
const isHealthy = ref(false)
let intervalId: ReturnType<typeof setInterval> | null = null

async function checkHealth() {
  try {
    const res = await fetch('/api/health')
    if (res.ok) {
      healthData.value = await res.json()
      isHealthy.value = true
    } else {
      isHealthy.value = false
    }
  } catch {
    isHealthy.value = false
  }
}

onMounted(() => {
  checkHealth()
  intervalId = setInterval(checkHealth, 30000)
})

onUnmounted(() => {
  if (intervalId) clearInterval(intervalId)
})

const pageTitle = computed(() => {
  return (route.meta?.title as string) ?? route.name?.toString() ?? ''
})
</script>

<template>
  <header
    class="fixed left-56 right-0 top-0 z-30 flex h-16 items-center justify-between border-b border-gray-200 bg-white px-8 shadow-sm"
  >
    <!-- Left: Page title -->
    <h1 class="text-xl font-semibold text-slate-800">{{ pageTitle }}</h1>

    <!-- Right: Status + Grafana -->
    <div class="flex items-center gap-4">
      <!-- Health status with tooltip -->
      <div class="group/health relative flex items-center gap-2">
        <StatusDot :healthy="isHealthy" label="API" />

        <!-- Tooltip -->
        <div
          class="pointer-events-none absolute right-0 top-full mt-2 w-64 rounded-lg border border-gray-200 bg-white p-3 opacity-0 shadow-lg transition-opacity duration-200 group-hover/health:pointer-events-auto group-hover/health:opacity-100"
        >
          <p class="mb-1 text-xs font-semibold text-slate-700">Backend Info</p>
          <template v-if="healthData">
            <div class="space-y-1 text-xs text-slate-600">
              <p v-if="healthData.version">
                <span class="font-medium">Version:</span> {{ healthData.version }}
              </p>
              <p v-if="healthData.gpu_name">
                <span class="font-medium">GPU:</span> {{ healthData.gpu_name }}
              </p>
              <p v-if="healthData.stations_count != null">
                <span class="font-medium">Stations:</span> {{ healthData.stations_count }}
              </p>
            </div>
          </template>
          <p v-else class="text-xs text-slate-400">Aucune donnée disponible</p>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { CheckCircle, XCircle, Loader, Clock } from 'lucide-vue-next'
import type { EvalStatus } from '../../api/types'

const props = defineProps<{
  status: EvalStatus | null
}>()

const percent = computed(() => {
  if (!props.status) return 0
  return Math.round(props.status.progress.percent)
})

const elapsedFormatted = computed(() => {
  if (!props.status) return '0s'
  const seconds = Math.round(props.status.progress.elapsed_seconds)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remaining = seconds % 60
  return `${minutes}m ${remaining}s`
})

const currentStep = computed(() => {
  if (!props.status) return ''
  return props.status.progress.current_step ?? ''
})

const stepsText = computed(() => {
  if (!props.status) return ''
  const p = props.status.progress
  if (p.steps_completed != null && p.steps_total != null) {
    return `${p.steps_completed} / ${p.steps_total}`
  }
  return ''
})

const isRunning = computed(
  () => props.status?.status === 'running' || props.status?.status === 'started',
)
const isCompleted = computed(() => props.status?.status === 'completed')
const isFailed = computed(() => props.status?.status === 'failed')
</script>

<template>
  <div class="rounded-lg border border-gray-200 bg-white p-5">
    <!-- Not started -->
    <div v-if="!status" class="flex items-center gap-3 text-sm text-gray-500">
      <Clock class="h-5 w-5" />
      <span>Aucune évaluation en cours</span>
    </div>

    <!-- Running -->
    <div v-else-if="isRunning" class="space-y-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Loader class="h-5 w-5 animate-spin text-blue-600" />
          <span class="text-sm font-medium text-gray-900">Évaluation en cours</span>
        </div>
        <div class="flex items-center gap-3 text-sm text-gray-500">
          <span v-if="stepsText">Étape {{ stepsText }}</span>
          <span>{{ elapsedFormatted }}</span>
        </div>
      </div>

      <!-- Progress bar -->
      <div class="h-2.5 w-full rounded-full bg-gray-200">
        <div
          class="h-2.5 rounded-full bg-blue-600 transition-all duration-300"
          :style="{ width: percent + '%' }"
        />
      </div>

      <div class="flex items-center justify-between text-sm">
        <span class="text-gray-600">{{ currentStep }}</span>
        <span class="font-medium text-blue-600">{{ percent }}%</span>
      </div>
    </div>

    <!-- Completed -->
    <div v-else-if="isCompleted" class="space-y-2">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <CheckCircle class="h-5 w-5 text-green-600" />
          <span class="text-sm font-medium text-green-700">Évaluation terminée</span>
        </div>
        <span class="text-sm text-gray-500">{{ elapsedFormatted }}</span>
      </div>
      <div class="h-2.5 w-full rounded-full bg-green-200">
        <div class="h-2.5 w-full rounded-full bg-green-500" />
      </div>
    </div>

    <!-- Failed -->
    <div v-else-if="isFailed" class="space-y-2">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <XCircle class="h-5 w-5 text-red-600" />
          <span class="text-sm font-medium text-red-700">Évaluation échouée</span>
        </div>
        <span class="text-sm text-gray-500">{{ elapsedFormatted }}</span>
      </div>
      <div class="h-2.5 w-full rounded-full bg-red-200">
        <div
          class="h-2.5 rounded-full bg-red-500 transition-all"
          :style="{ width: percent + '%' }"
        />
      </div>
      <p v-if="currentStep" class="text-sm text-red-600">{{ currentStep }}</p>
    </div>
  </div>
</template>

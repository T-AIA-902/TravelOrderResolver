<script setup lang="ts">
import { Mic, Loader2 } from 'lucide-vue-next'
import type { RecorderState } from '../../composables/useAudioRecorder'

defineProps<{
  state: RecorderState
  disabled?: boolean
}>()

defineEmits<{
  toggle: []
}>()
</script>

<template>
  <button
    type="button"
    :disabled="disabled || state === 'processing'"
    class="relative inline-flex items-center justify-center rounded-xl p-3.5 transition-colors focus:outline-none focus:ring-2 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
    :class="{
      'bg-gray-100 text-gray-500 hover:bg-blue-50 hover:text-blue-600 focus:ring-blue-400':
        state === 'idle',
      'bg-red-100 text-red-600 focus:ring-red-400': state === 'recording',
      'bg-gray-100 text-gray-400 focus:ring-gray-400': state === 'processing',
    }"
    @click="$emit('toggle')"
  >
    <!-- Pulse ring when recording -->
    <span
      v-if="state === 'recording'"
      class="absolute inset-0 animate-ping rounded-xl bg-red-200 opacity-40"
    />

    <!-- Icon -->
    <Loader2 v-if="state === 'processing'" class="h-5 w-5 animate-spin" />
    <Mic v-else class="relative h-5 w-5" />
  </button>
</template>

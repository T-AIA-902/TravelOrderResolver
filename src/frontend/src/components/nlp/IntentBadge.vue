<script setup lang="ts">
import type { Intent } from '../../api/types'

const props = defineProps<{
  intent: Intent
  confidence: number
}>()

const badgeClasses: Record<Intent, string> = {
  TRIP: 'bg-green-100 text-green-800',
  NOT_TRIP: 'bg-gray-100 text-gray-800',
  UNKNOWN: 'bg-yellow-100 text-yellow-800',
}

const intentLabels: Record<Intent, string> = {
  TRIP: 'Voyage',
  NOT_TRIP: 'Non-voyage',
  UNKNOWN: 'Inconnu',
}
</script>

<template>
  <span
    class="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium"
    :class="badgeClasses[props.intent]"
  >
    <span>{{ intentLabels[props.intent] }}</span>
    <span class="opacity-70">({{ Math.round(props.confidence * 100) }}%)</span>
  </span>
</template>

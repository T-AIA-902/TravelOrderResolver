<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    label: string
    value: number
    max?: number
    unit?: string
    color?: string
  }>(),
  {
    max: 100,
    unit: '%',
    color: 'bg-blue-500',
  },
)

const percent = computed(() => Math.min(100, (props.value / props.max) * 100))
const barColor = computed(() => {
  if (percent.value > 90) return 'bg-red-500'
  if (percent.value > 70) return 'bg-amber-500'
  return props.color
})
</script>

<template>
  <div>
    <div class="mb-1 flex items-center justify-between text-sm">
      <span class="font-medium text-gray-700">{{ label }}</span>
      <span class="text-gray-500">{{ value.toFixed(1) }}{{ unit }} / {{ max }}{{ unit }}</span>
    </div>
    <div class="h-2.5 w-full overflow-hidden rounded-full bg-gray-200">
      <div
        class="h-full rounded-full transition-all duration-500"
        :class="barColor"
        :style="{ width: `${percent}%` }"
      />
    </div>
  </div>
</template>

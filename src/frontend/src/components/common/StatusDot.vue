<script setup lang="ts">
defineProps<{
  healthy: boolean
  label?: string
}>()
</script>

<template>
  <div class="flex items-center gap-2">
    <span class="relative flex h-3 w-3">
      <span
        v-if="healthy"
        class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"
      />
      <span
        class="relative inline-flex rounded-full h-3 w-3"
        :class="healthy ? 'bg-green-500' : 'bg-red-500'"
      />
    </span>
    <span v-if="label" class="text-sm text-gray-600">{{ label }}</span>
  </div>
</template>

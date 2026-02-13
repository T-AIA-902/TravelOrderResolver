<script setup lang="ts">
defineProps<{
  size?: 'sm' | 'md' | 'lg'
}>()
</script>

<template>
  <div
    class="animate-spin rounded-full border-2 border-gray-300 border-t-blue-600"
    :class="{
      'h-4 w-4': size === 'sm' || !size,
      'h-6 w-6': size === 'md',
      'h-8 w-8': size === 'lg',
    }"
  />
</template>

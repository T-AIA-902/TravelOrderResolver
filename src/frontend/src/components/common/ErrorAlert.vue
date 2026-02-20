<script setup lang="ts">
defineProps<{
  message: string
}>()
</script>

<template>
  <div class="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
    <p>{{ message }}</p>
  </div>
</template>

<script setup lang="ts">
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <AppSidebar />
    <AppHeader />

    <!-- Main content area -->
    <main class="ml-16 mt-14 min-h-screen bg-gray-50 p-6">
      <slot />
    </main>
  </div>
</template>

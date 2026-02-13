<script setup lang="ts">
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'
</script>

<template>
  <div class="min-h-screen bg-gray-50">
    <AppSidebar />
    <AppHeader />

    <!-- Main content area -->
    <main class="ml-16 mt-14 h-[calc(100vh-3.5rem)] overflow-auto bg-gray-50">
      <slot />
    </main>
  </div>
</template>

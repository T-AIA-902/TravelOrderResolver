<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppSidebar from './AppSidebar.vue'
import AppHeader from './AppHeader.vue'

const route = useRoute()
const isHome = computed(() => route.path === '/')
</script>

<template>
  <div class="h-screen overflow-hidden bg-gray-50">
    <AppSidebar v-if="!isHome" />
    <AppHeader v-if="!isHome" />

    <!-- Main content area -->
    <main :class="isHome ? 'h-full overflow-auto' : 'ml-56 pt-16 h-full overflow-auto'">
      <slot />
    </main>
  </div>
</template>

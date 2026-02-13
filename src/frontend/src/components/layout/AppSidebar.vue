<script setup lang="ts">
import { useRoute } from 'vue-router'
import { LayoutDashboard, MessageSquare, BarChart3 } from 'lucide-vue-next'

const route = useRoute()

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/chat', label: 'Chat', icon: MessageSquare },
  { to: '/evaluation', label: 'Évaluation', icon: BarChart3 },
]

function isActive(to: string): boolean {
  if (to === '/') return route.path === '/'
  return route.path.startsWith(to)
}
</script>

<template>
  <aside
    class="group fixed left-0 top-0 z-40 flex h-screen w-16 flex-col bg-slate-900 text-white transition-all duration-300 ease-in-out hover:w-56"
  >
    <!-- Logo / Title -->
    <router-link to="/" class="flex h-14 items-center gap-3 border-b border-slate-700 px-4">
      <span class="text-lg font-bold tracking-wider">TOR</span>
      <span
        class="whitespace-nowrap text-sm font-medium opacity-0 transition-opacity duration-300 group-hover:opacity-100"
      >
        Travel Order Resolver
      </span>
    </router-link>

    <!-- Navigation -->
    <nav class="mt-4 flex flex-1 flex-col gap-1 px-2">
      <router-link
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        class="flex items-center gap-3 rounded px-3 py-2.5 text-slate-300 transition-colors duration-200 hover:bg-slate-800 hover:text-white"
        :class="{ 'bg-slate-700 text-white': isActive(item.to) }"
      >
        <component :is="item.icon" class="h-5 w-5 shrink-0" />
        <span
          class="whitespace-nowrap text-sm opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        >
          {{ item.label }}
        </span>
      </router-link>
    </nav>
  </aside>
</template>

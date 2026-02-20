<script setup lang="ts">
import { useRoute } from 'vue-router'
import {
  Home,
  MessageSquare,
  BarChart3,
  FileText,
  Database,
  Activity,
  Train,
} from 'lucide-vue-next'

const route = useRoute()

const navItems = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/chat', label: 'Chat', icon: MessageSquare },
  { to: '/evaluation', label: 'Évaluation', icon: BarChart3 },
  { to: '/dataset', label: 'Dataset', icon: Database },
  { to: '/rapports', label: 'Rapports', icon: FileText },
  { to: '/monitoring', label: 'Monitoring', icon: Activity },
]

function isActive(to: string): boolean {
  if (to === '/') return route.path === '/'
  return route.path.startsWith(to)
}
</script>

<template>
  <aside class="fixed left-0 top-0 z-40 flex h-screen w-56 flex-col bg-slate-900 text-white">
    <!-- Logo -->
    <router-link to="/" class="flex h-16 items-center gap-3 border-b border-slate-700/50 px-5">
      <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-800">
        <Train class="h-5 w-5 text-white" />
      </div>
      <span class="text-sm font-semibold tracking-wide">Travel Order Resolver</span>
    </router-link>

    <!-- Navigation -->
    <nav class="mt-6 flex flex-1 flex-col gap-1 px-3">
      <router-link
        v-for="item in navItems"
        :key="item.to"
        :to="item.to"
        class="flex items-center gap-3 rounded-lg px-3 py-3 text-slate-400 transition-colors duration-150 hover:bg-slate-800 hover:text-white"
        :class="{ 'bg-slate-800 text-white font-medium': isActive(item.to) }"
      >
        <component :is="item.icon" class="h-5 w-5 shrink-0" />
        <span class="text-sm">{{ item.label }}</span>
      </router-link>
    </nav>

    <!-- Footer -->
    <div class="border-t border-slate-700/50 px-5 py-4">
      <p class="text-xs text-slate-500">T-AIA-902 — Epitech</p>
    </div>
  </aside>
</template>

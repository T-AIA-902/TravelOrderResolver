import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      meta: { title: 'Dashboard' },
      component: () => import('../pages/DashboardPage.vue'),
    },
    {
      path: '/chat',
      name: 'chat',
      meta: { title: 'Chat' },
      component: () => import('../pages/ChatPage.vue'),
    },
    {
      path: '/evaluation',
      name: 'evaluation',
      meta: { title: 'Évaluation' },
      component: () => import('../pages/EvaluationPage.vue'),
    },
    {
      path: '/map',
      name: 'map-explorer',
      meta: { title: 'Carte' },
      component: () => import('../pages/MapExplorerPage.vue'),
    },
  ],
})

export default router

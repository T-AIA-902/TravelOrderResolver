import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'home',
      meta: { title: 'Home' },
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
      path: '/rapports',
      name: 'rapports',
      meta: { title: 'Rapports' },
      component: () => import('../pages/RapportsPage.vue'),
    },
    {
      path: '/dataset',
      name: 'dataset',
      meta: { title: 'Dataset' },
      component: () => import('../pages/DatasetPage.vue'),
    },
  ],
})

export default router

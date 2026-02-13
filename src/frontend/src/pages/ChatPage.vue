<script setup lang="ts">
import { ref, computed } from 'vue'
import type { NlpResult, PathfindingResult } from '../api/types'
import { useResolve } from '../composables/useResolve'
import { useMapRoute } from '../composables/useMapRoute'
import ChatWindow from '../components/chat/ChatWindow.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import RailwayMap from '../components/map/RailwayMap.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import { MessageCircle } from 'lucide-vue-next'

interface ChatMessageItem {
  id: string
  role: 'user' | 'system'
  text: string
  nlp?: NlpResult
  pathfinding?: PathfindingResult
}

const messages = ref<ChatMessageItem[]>([])
const { isLoading, error, resolveTrip } = useResolve()
const {
  routeSegments,
  departureStation,
  destinationStation,
  intermediateStations,
  updateFromResponse,
} = useMapRoute()

let msgCounter = 0
function nextId() {
  return `msg-${++msgCounter}`
}

async function onSend(text: string) {
  messages.value.push({
    id: nextId(),
    role: 'user',
    text,
  })

  try {
    const response = await resolveTrip({ text })

    const systemMsg: ChatMessageItem = {
      id: nextId(),
      role: 'system',
      text: response.nlp.intent.value === 'TRIP' ? "Résultat de l'analyse :" : 'Requête analysée :',
      nlp: response.nlp,
      pathfinding: response.pathfinding ?? undefined,
    }
    messages.value.push(systemMsg)

    if (response.pathfinding?.found) {
      updateFromResponse(response)
    }
  } catch {
    messages.value.push({
      id: nextId(),
      role: 'system',
      text: error.value ?? 'Erreur lors du traitement de la requête',
    })
  }
}

const showDebug = ref(false)
const isEmpty = computed(() => messages.value.length === 0 && !isLoading.value)
</script>

<template>
  <div class="flex h-full overflow-hidden">
    <!-- Left: Chat (60%) -->
    <div class="flex w-3/5 flex-col border-r border-gray-200">
      <!-- Empty state: centered input -->
      <div v-if="isEmpty" class="flex flex-1 flex-col items-center justify-center px-4">
        <MessageCircle class="mb-5 h-16 w-16 text-gray-300" />
        <h2 class="mb-2 text-2xl font-semibold text-gray-700">Travel Order Resolver</h2>
        <p class="mb-8 text-base text-gray-400">Posez une question pour commencer...</p>
        <div class="w-full max-w-2xl [&>div]:border-t-0">
          <ChatInput :disabled="isLoading" @send="onSend" />
        </div>
      </div>

      <!-- Conversation state: messages + input at bottom -->
      <template v-else>
        <ChatWindow :messages="messages" :loading="isLoading" />
        <ChatInput :disabled="isLoading" @send="onSend" />
      </template>
    </div>

    <!-- Right: Map + Debug (40%) -->
    <div class="relative z-0 flex w-2/5 flex-col overflow-hidden">
      <div class="min-h-0 flex-1 p-4">
        <RailwayMap
          :route="routeSegments"
          :departure-station="departureStation"
          :destination-station="destinationStation"
          :intermediate-stations="intermediateStations"
          height="100%"
        />
      </div>

      <!-- Debug panel -->
      <div class="border-t border-gray-200">
        <button
          class="flex w-full items-center justify-between px-4 py-2 text-xs font-medium text-gray-500 hover:bg-gray-50"
          @click="showDebug = !showDebug"
        >
          <span>NLP Debug JSON</span>
          <span>{{ showDebug ? '▲' : '▼' }}</span>
        </button>
        <div v-if="showDebug" class="max-h-48 overflow-y-auto bg-gray-50 px-4 py-2">
          <pre class="whitespace-pre-wrap text-xs text-gray-600">{{
            (() => {
              const last = messages.filter((m) => m.nlp).slice(-1)[0]
              return last?.nlp ? JSON.stringify(last.nlp, null, 2) : 'Aucune donnée NLP'
            })()
          }}</pre>
        </div>
      </div>

      <ErrorAlert v-if="error" :message="error" />
    </div>
  </div>
</template>

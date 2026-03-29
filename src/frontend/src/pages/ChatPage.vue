<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import type { NlpResult, PathfindingResult } from '../api/types'
import { useResolve } from '../composables/useResolve'
import { useMapRoute } from '../composables/useMapRoute'
import { useEvaluation } from '../composables/useEvaluation'
import ChatWindow from '../components/chat/ChatWindow.vue'
import ChatInput from '../components/chat/ChatInput.vue'
import RailwayMap from '../components/map/RailwayMap.vue'
import ErrorAlert from '../components/common/ErrorAlert.vue'
import { MessageCircle, Settings2 } from 'lucide-vue-next'

interface ChatMessageItem {
  id: string
  role: 'user' | 'system'
  text: string
  nlp?: NlpResult
  pathfinding?: PathfindingResult
}

const messages = ref<ChatMessageItem[]>([])
const { isLoading, error, resolveTrip } = useResolve()

const intentModels = ['CamemBERT', 'SpaCy', 'Flan-T5', 'Regex'] as const
const entityModels = ['Flan-T5', 'CamemBERT', 'SpaCy', 'Regex'] as const

const selectedIntentModel = ref('CamemBERT')
const selectedEntityModel = ref('Flan-T5')
const useFuzzy = ref(true)

// Load best combo from latest evaluation
const { reports, reportDetail, loadReports, loadReport } = useEvaluation()

onMounted(async () => {
  await loadReports()
  if (reports.value.length > 0) {
    await loadReport(reports.value[0].id)
    if (reportDetail.value) {
      const data = reportDetail.value as Record<string, unknown>
      const combined = data.combined_fuzzy as Array<Record<string, unknown>> | undefined
      if (combined && combined.length > 0) {
        // Find best combined pipeline
        let best = combined[0]
        for (const entry of combined) {
          const score = (entry.intent_accuracy as number) + (entry.entity_accuracy as number)
          const bestScore = (best.intent_accuracy as number) + (best.entity_accuracy as number)
          if (score > bestScore) best = entry
        }
        selectedIntentModel.value = best.intent as string
        selectedEntityModel.value = best.entity as string
      }
    }
  }
})

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
    const response = await resolveTrip({
      text,
      intent_model: selectedIntentModel.value,
      entity_model: selectedEntityModel.value,
      use_fuzzy: useFuzzy.value,
    })

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
const showSettings = ref(true)
const isEmpty = computed(() => messages.value.length === 0 && !isLoading.value)

const pipelineLabel = computed(() => {
  return `${selectedIntentModel.value} + ${selectedEntityModel.value}${useFuzzy.value ? ' + Fuzzy' : ''}`
})
</script>

<template>
  <div class="flex h-full overflow-hidden">
    <!-- Left: Chat -->
    <div class="flex w-1/2 flex-col border-r border-gray-200">
      <!-- Empty state: centered input -->
      <div v-if="isEmpty" class="flex flex-1 flex-col items-center justify-center px-4">
        <MessageCircle class="mb-5 h-16 w-16 text-gray-300" />
        <h2 class="mb-2 text-2xl font-semibold text-gray-700">Travel Order Resolver</h2>
        <p class="mb-2 text-base text-gray-400">Posez une question pour commencer...</p>
        <p class="mb-8 text-xs text-gray-400">
          Pipeline : <span class="rounded bg-blue-50 px-1.5 py-0.5 font-medium text-blue-700">{{ pipelineLabel }}</span>
        </p>
        <div class="w-full max-w-2xl [&>div]:border-t-0">
          <ChatInput :disabled="isLoading" @send="onSend" />
        </div>
      </div>

      <!-- Conversation state: messages + input at bottom -->
      <template v-else>
        <ChatWindow :messages="messages" :loading="isLoading" />
        <ChatInput :disabled="isLoading" @send="onSend" />
      </template>

      <!-- Model selector bar -->
      <div class="border-t border-gray-100 bg-gray-50 px-4 py-2">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1">
            <button
              class="flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-gray-500 transition hover:bg-gray-200"
              @click="showSettings = !showSettings"
            >
              <Settings2 class="h-3.5 w-3.5" />
              Pipeline
            </button>
            <span class="text-xs text-gray-400">{{ pipelineLabel }}</span>
          </div>
        </div>

        <div v-if="showSettings" class="mt-2 flex flex-wrap items-center gap-4">
          <div class="flex items-center gap-1.5">
            <label class="text-xs font-medium text-gray-500">Intent :</label>
            <select
              v-model="selectedIntentModel"
              class="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
            >
              <option v-for="m in intentModels" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>

          <div class="flex items-center gap-1.5">
            <label class="text-xs font-medium text-gray-500">Entity :</label>
            <select
              v-model="selectedEntityModel"
              class="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 focus:border-blue-400 focus:outline-none focus:ring-1 focus:ring-blue-400"
            >
              <option v-for="m in entityModels" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>

          <label class="flex items-center gap-1.5 text-xs text-gray-500">
            <input
              v-model="useFuzzy"
              type="checkbox"
              class="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            Fuzzy matching
          </label>
        </div>
      </div>
    </div>

    <!-- Right: Map + Debug -->
    <div class="relative z-0 flex w-1/2 flex-col overflow-hidden">
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

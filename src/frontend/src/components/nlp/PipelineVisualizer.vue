<script setup lang="ts">
import { computed } from 'vue'
import type { NlpResult, Language } from '../../api/types'
import IntentBadge from './IntentBadge.vue'
import EntityHighlight from './EntityHighlight.vue'

const props = defineProps<{
  nlp: NlpResult
}>()

const isNotTrip = computed(() => props.nlp.intent.value === 'NOT_TRIP')

const languageFlags: Record<Language, string> = {
  FRENCH: '\u{1F1EB}\u{1F1F7}',
  ENGLISH: '\u{1F1EC}\u{1F1E7}',
  UNKNOWN: '\u2753',
}

const languageLabels: Record<Language, string> = {
  FRENCH: 'Français',
  ENGLISH: 'Anglais',
  UNKNOWN: 'Inconnue',
}
</script>

<template>
  <div class="space-y-0">
    <!-- Step 1: Langue détectée -->
    <div class="relative flex gap-4">
      <!-- Connector line -->
      <div class="flex flex-col items-center">
        <div
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-sm font-bold text-indigo-700"
        >
          1
        </div>
        <div class="w-0.5 grow bg-gray-200" />
      </div>
      <!-- Content -->
      <div class="mb-6 w-full rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h4 class="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
          Langue détectée
        </h4>
        <div class="flex items-center gap-2">
          <span class="text-lg">{{ languageFlags[props.nlp.language.detected] }}</span>
          <span class="text-sm font-medium text-gray-700">
            {{ languageLabels[props.nlp.language.detected] }}
          </span>
          <span
            class="inline-flex rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-600"
          >
            {{ Math.round(props.nlp.language.confidence * 100) }}%
          </span>
        </div>
      </div>
    </div>

    <!-- Step 2: Classification intent -->
    <div class="relative flex gap-4">
      <div class="flex flex-col items-center">
        <div
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-sm font-bold text-indigo-700"
        >
          2
        </div>
        <div class="w-0.5 grow bg-gray-200" />
      </div>
      <div class="mb-6 w-full rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h4 class="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
          Classification intent
        </h4>
        <IntentBadge :intent="props.nlp.intent.value" :confidence="props.nlp.intent.confidence" />
      </div>
    </div>

    <!-- Step 3: Entités extraites -->
    <div class="relative flex gap-4">
      <div class="flex flex-col items-center">
        <div
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-sm font-bold text-indigo-700"
        >
          3
        </div>
        <div class="w-0.5 grow bg-gray-200" />
      </div>
      <div class="mb-6 w-full rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h4 class="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
          Entités extraites
        </h4>
        <template v-if="isNotTrip">
          <p class="text-sm text-gray-400 italic">
            Non-voyage détecté — pas d'extraction d'entités
          </p>
        </template>
        <template v-else>
          <EntityHighlight :entities="props.nlp.entities" />
        </template>
      </div>
    </div>

    <!-- Step 4: Latence -->
    <div class="relative flex gap-4">
      <div class="flex flex-col items-center">
        <div
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-sm font-bold text-indigo-700"
        >
          4
        </div>
        <!-- No trailing line for last step -->
      </div>
      <div class="w-full rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
        <h4 class="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">Latence</h4>
        <span
          class="inline-flex rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600"
        >
          Pipeline : {{ props.nlp.latency_ms }}ms
        </span>
      </div>
    </div>
  </div>
</template>

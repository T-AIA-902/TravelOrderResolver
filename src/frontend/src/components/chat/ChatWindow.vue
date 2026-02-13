<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import type { NlpResult, PathfindingResult } from '../../api/types'
import ChatMessage from './ChatMessage.vue'

interface ChatMessageItem {
  id: string
  role: 'user' | 'system'
  text: string
  nlp?: NlpResult
  pathfinding?: PathfindingResult
}

const props = defineProps<{
  messages: ChatMessageItem[]
  loading: boolean
}>()

const scrollContainer = ref<HTMLElement | null>(null)

function scrollToBottom() {
  nextTick(() => {
    if (scrollContainer.value) {
      scrollContainer.value.scrollTo({
        top: scrollContainer.value.scrollHeight,
        behavior: 'smooth',
      })
    }
  })
}

watch(
  () => props.messages.length,
  () => scrollToBottom(),
)

watch(
  () => props.loading,
  () => scrollToBottom(),
)
</script>

<template>
  <div ref="scrollContainer" class="flex-1 overflow-y-auto px-4 py-6">
    <!-- Messages -->
    <div class="mx-auto flex max-w-3xl flex-col gap-4">
      <ChatMessage
        v-for="msg in messages"
        :key="msg.id"
        :role="msg.role"
        :text="msg.text"
        :nlp="msg.nlp"
        :pathfinding="msg.pathfinding"
      />

      <!-- Typing indicator -->
      <div v-if="loading" class="flex justify-start">
        <div class="flex items-center gap-1 rounded-xl border border-gray-200 bg-white px-4 py-3">
          <span class="typing-dot h-2 w-2 rounded-full bg-gray-400" />
          <span class="typing-dot animation-delay-200 h-2 w-2 rounded-full bg-gray-400" />
          <span class="typing-dot animation-delay-400 h-2 w-2 rounded-full bg-gray-400" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.typing-dot {
  animation: typing-bounce 1.4s infinite ease-in-out both;
}

.animation-delay-200 {
  animation-delay: 0.2s;
}

.animation-delay-400 {
  animation-delay: 0.4s;
}

@keyframes typing-bounce {
  0%,
  80%,
  100% {
    transform: scale(0.6);
    opacity: 0.4;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}
</style>

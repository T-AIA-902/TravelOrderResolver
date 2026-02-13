<script setup lang="ts">
import { ref } from 'vue'
import { Send } from 'lucide-vue-next'

defineProps<{
  disabled: boolean
}>()

const emit = defineEmits<{
  send: [text: string]
}>()

const text = ref('')

function handleSend() {
  const trimmed = text.value.trim()
  if (!trimmed) return
  emit('send', trimmed)
  text.value = ''
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<template>
  <div class="border-t border-gray-200 bg-white px-6 py-4">
    <div class="flex items-center gap-3">
      <input
        v-model="text"
        type="text"
        :disabled="disabled"
        placeholder="Ex : Je veux aller de Paris à Lyon via Dijon"
        class="flex-1 rounded-xl border border-gray-200 bg-gray-50 px-5 py-3.5 text-base text-gray-900 placeholder-gray-400 focus:border-blue-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-blue-400 disabled:opacity-50"
        @keydown="onKeydown"
      />
      <button
        type="button"
        :disabled="disabled || !text.trim()"
        class="inline-flex items-center justify-center rounded-xl bg-blue-600 px-5 py-3.5 text-white transition-colors hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
        @click="handleSend"
      >
        <Send class="h-5 w-5" />
      </button>
    </div>
  </div>
</template>

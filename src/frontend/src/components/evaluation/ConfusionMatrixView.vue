<script setup lang="ts">
import { computed } from 'vue'
import type { ConfusionMatrix } from '../../api/types'

const props = defineProps<{
  matrix: ConfusionMatrix
}>()

const maxValue = computed(() => {
  let max = 0
  for (const row of props.matrix.matrix) {
    for (const cell of row) {
      if (cell > max) max = cell
    }
  }
  return max
})

function cellOpacity(value: number): number {
  if (maxValue.value === 0) return 0
  return value / maxValue.value
}

function cellStyle(value: number): Record<string, string> {
  const opacity = cellOpacity(value)
  return {
    backgroundColor: `rgba(59, 130, 246, ${opacity})`,
    color: opacity > 0.5 ? 'white' : '#1f2937',
  }
}
</script>

<template>
  <div class="overflow-x-auto">
    <table class="text-sm">
      <thead>
        <tr>
          <th
            class="px-2 py-2 text-xs font-medium text-gray-500"
            :title="'Prédit (colonnes) vs Réel (lignes)'"
          >
            Réel \ Prédit
          </th>
          <th
            v-for="label in matrix.labels"
            :key="'h-' + label"
            class="px-2 py-2 text-center text-xs font-medium text-gray-600"
          >
            {{ label }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(row, rowIdx) in matrix.matrix" :key="'r-' + rowIdx">
          <td class="px-2 py-2 text-xs font-medium text-gray-600">
            {{ matrix.labels[rowIdx] }}
          </td>
          <td
            v-for="(cell, colIdx) in row"
            :key="'c-' + colIdx"
            class="min-w-[3rem] px-2 py-2 text-center text-xs font-medium"
            :style="cellStyle(cell)"
          >
            {{ cell }}
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

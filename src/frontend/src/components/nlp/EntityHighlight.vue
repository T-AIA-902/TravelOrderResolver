<script setup lang="ts">
import { computed } from 'vue'
import type { EntitiesResult, EntityMatch } from '../../api/types'

const props = defineProps<{
  entities: EntitiesResult
}>()

const hasEntities = computed(
  () => props.entities.departure !== null || props.entities.destination !== null,
)

function formatEntity(entity: EntityMatch): string {
  if (entity.matched) {
    return `${entity.raw} \u2192 ${entity.matched}`
  }
  return `${entity.raw} \u2192 Non trouvé`
}
</script>

<template>
  <div class="space-y-2">
    <template v-if="hasEntities">
      <!-- Départ -->
      <div v-if="props.entities.departure" class="flex items-start gap-2">
        <span class="shrink-0 text-sm font-medium text-gray-500">Départ :</span>
        <span
          class="inline-flex rounded-md border px-2.5 py-0.5 text-sm font-medium bg-green-100 text-green-800 border-green-300"
        >
          {{ formatEntity(props.entities.departure) }}
        </span>
      </div>

      <!-- Destination -->
      <div v-if="props.entities.destination" class="flex items-start gap-2">
        <span class="shrink-0 text-sm font-medium text-gray-500">Destination :</span>
        <span
          class="inline-flex rounded-md border px-2.5 py-0.5 text-sm font-medium bg-red-100 text-red-800 border-red-300"
        >
          {{ formatEntity(props.entities.destination) }}
        </span>
      </div>

      <!-- Intermédiaires -->
      <div v-if="props.entities.intermediates.length > 0" class="flex items-start gap-2">
        <span class="shrink-0 text-sm font-medium text-gray-500">Intermédiaires :</span>
        <div class="flex flex-wrap gap-1.5">
          <span
            v-for="(entity, index) in props.entities.intermediates"
            :key="index"
            class="inline-flex rounded-md border px-2.5 py-0.5 text-sm font-medium bg-blue-100 text-blue-800 border-blue-300"
          >
            {{ formatEntity(entity) }}
          </span>
        </div>
      </div>
    </template>

    <template v-else>
      <p class="text-sm text-gray-400 italic">Aucune entité détectée</p>
    </template>
  </div>
</template>

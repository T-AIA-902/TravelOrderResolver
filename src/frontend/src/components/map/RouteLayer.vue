<script setup lang="ts">
import { LPolyline, LPopup } from '@vue-leaflet/vue-leaflet'
import type { RouteSegment } from '../../api/types'

defineProps<{
  segments: RouteSegment[]
}>()

function polylineOptions(segment: RouteSegment) {
  if (segment.type === 'TRAIN') {
    return {
      color: '#ef4444',
      weight: 4,
      opacity: 0.8,
    }
  }
  return {
    color: '#3b82f6',
    weight: 3,
    opacity: 0.7,
    dashArray: '8,6',
  }
}

function popupText(segment: RouteSegment): string {
  if (segment.type === 'TRAIN') {
    return `Ligne ${segment.line}`
  }
  return 'Correspondance à pied'
}
</script>

<template>
  <LPolyline
    v-for="(segment, index) in segments"
    :key="index"
    :lat-lngs="segment.geometry"
    v-bind="polylineOptions(segment)"
  >
    <LPopup>
      <span class="text-sm">{{ popupText(segment) }}</span>
    </LPopup>
  </LPolyline>
</template>

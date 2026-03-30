<script setup lang="ts">
import { LPolyline, LPopup } from '@vue-leaflet/vue-leaflet'
import type { RouteSegment } from '../../api/types'

const props = withDefaults(
  defineProps<{
    segments: RouteSegment[]
    color?: string
    opacity?: number
    label?: string
  }>(),
  {
    color: '#ef4444',
    opacity: 0.8,
    label: '',
  },
)

function polylineOptions(segment: RouteSegment) {
  if (segment.type === 'TRAIN') {
    return {
      color: props.color,
      weight: props.opacity >= 0.7 ? 4 : 3,
      opacity: props.opacity,
    }
  }
  return {
    color: props.color,
    weight: 3,
    opacity: props.opacity * 0.8,
    dashArray: '8,6',
  }
}

function popupText(segment: RouteSegment): string {
  const prefix = props.label ? `[${props.label}] ` : ''
  if (segment.type === 'TRAIN') {
    return `${prefix}Ligne ${segment.line}`
  }
  return `${prefix}Correspondance à pied`
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

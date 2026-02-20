<script setup lang="ts">
import { computed } from 'vue'
import { LMarker, LPopup } from '@vue-leaflet/vue-leaflet'
import L from 'leaflet'
import type { Station } from '../../api/types'

const props = withDefaults(
  defineProps<{
    station: Station
    type?: 'departure' | 'destination' | 'intermediate'
  }>(),
  {
    type: 'intermediate',
  },
)

const colorMap: Record<string, string> = {
  departure: '#22c55e',
  destination: '#ef4444',
  intermediate: '#3b82f6',
}

const icon = computed(() =>
  L.divIcon({
    className: '',
    html: `<div style="
      width: 16px;
      height: 16px;
      border-radius: 9999px;
      border: 2px solid white;
      background-color: ${colorMap[props.type]};
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    "></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    popupAnchor: [0, -10],
  }),
)
</script>

<template>
  <LMarker :lat-lng="[station.lat, station.lon]" :icon="icon as any">
    <LPopup>
      <div class="text-sm">
        <p class="font-semibold">{{ station.name }}</p>
        <p class="text-xs text-gray-500">UIC: {{ station.uic }}</p>
      </div>
    </LPopup>
  </LMarker>
</template>

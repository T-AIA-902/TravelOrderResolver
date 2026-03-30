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

const config: Record<string, { color: string; size: number; symbol: string }> = {
  departure: { color: '#22c55e', size: 24, symbol: '▶' },
  destination: { color: '#ef4444', size: 24, symbol: '◉' },
  intermediate: { color: '#f59e0b', size: 14, symbol: '' },
}

const icon = computed(() => {
  const c = config[props.type]
  const s = c.size

  if (props.type === 'intermediate') {
    // Small diamond for transfers
    return L.divIcon({
      className: '',
      html: `<div style="
        width: ${s}px;
        height: ${s}px;
        border-radius: 3px;
        transform: rotate(45deg);
        border: 2px solid white;
        background-color: ${c.color};
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      "></div>`,
      iconSize: [s, s],
      iconAnchor: [s / 2, s / 2],
      popupAnchor: [0, -s / 2],
    })
  }

  // Larger circle with symbol for departure/destination
  return L.divIcon({
    className: '',
    html: `<div style="
      width: ${s}px;
      height: ${s}px;
      border-radius: 9999px;
      border: 3px solid white;
      background-color: ${c.color};
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 12px;
      font-weight: bold;
    ">${c.symbol}</div>`,
    iconSize: [s, s],
    iconAnchor: [s / 2, s / 2],
    popupAnchor: [0, -s / 2],
  })
})

const labelMap: Record<string, string> = {
  departure: 'Départ',
  destination: 'Arrivée',
  intermediate: 'Correspondance',
}
</script>

<template>
  <LMarker :lat-lng="[station.lat, station.lon]" :icon="icon as any">
    <LPopup>
      <div class="text-sm">
        <p class="text-xs font-semibold uppercase tracking-wide" :style="{ color: config[type].color }">
          {{ labelMap[type] }}
        </p>
        <p class="font-semibold">{{ station.name }}</p>
      </div>
    </LPopup>
  </LMarker>
</template>

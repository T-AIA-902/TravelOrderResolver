<script setup lang="ts">
import { ref, watch } from 'vue'
import { LMap, LTileLayer } from '@vue-leaflet/vue-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { RouteSegment, Station } from '../../api/types'
import RouteLayer from './RouteLayer.vue'
import StationMarker from './StationMarker.vue'

const props = withDefaults(
  defineProps<{
    route?: RouteSegment[] | null
    departureStation?: Station | null
    destinationStation?: Station | null
    intermediateStations?: Station[]
    height?: string
  }>(),
  {
    route: null,
    departureStation: null,
    destinationStation: null,
    intermediateStations: () => [],
    height: '100%',
  },
)

const mapRef = ref<InstanceType<typeof LMap> | null>(null)

const center = ref<[number, number]>([46.603354, 1.888334])
const zoom = ref(6)

const tileUrl = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
const tileAttribution =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CartoDB</a>'

watch(
  () => props.route,
  (segments) => {
    if (!segments || segments.length === 0 || !mapRef.value) return

    const allPoints: [number, number][] = []
    for (const segment of segments) {
      for (const point of segment.geometry) {
        allPoints.push(point)
      }
    }

    if (allPoints.length === 0) return

    const bounds = L.latLngBounds(allPoints.map((p) => L.latLng(p[0], p[1])))
    const leafletMap = (mapRef.value as unknown as { leafletObject: L.Map }).leafletObject
    if (leafletMap) {
      leafletMap.fitBounds(bounds, { padding: [50, 50] })
    }
  },
  { deep: true },
)
</script>

<template>
  <div
    class="w-full overflow-hidden rounded-xl border border-gray-200"
    :style="{ height: props.height }"
  >
    <LMap
      ref="mapRef"
      :center="center"
      :zoom="zoom"
      :use-global-leaflet="false"
      class="h-full w-full"
    >
      <LTileLayer :url="tileUrl" :attribution="tileAttribution" />

      <RouteLayer v-if="route" :segments="route" />

      <StationMarker v-if="departureStation" :station="departureStation" type="departure" />
      <StationMarker v-if="destinationStation" :station="destinationStation" type="destination" />
      <StationMarker
        v-for="station in intermediateStations"
        :key="station.uic"
        :station="station"
        type="intermediate"
      />
    </LMap>
  </div>
</template>

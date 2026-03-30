<script setup lang="ts">
import { ref, watch } from 'vue'
import { LMap, LTileLayer } from '@vue-leaflet/vue-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { RouteSegment, Station } from '../../api/types'
import RouteLayer from './RouteLayer.vue'
import StationMarker from './StationMarker.vue'

export interface MapRoute {
  segments: RouteSegment[]
  color: string
  label: string
}

const props = withDefaults(
  defineProps<{
    route?: RouteSegment[] | null
    routes?: MapRoute[]
    selectedRouteIndex?: number
    departureStation?: Station | null
    destinationStation?: Station | null
    intermediateStations?: Station[]
    height?: string
  }>(),
  {
    route: null,
    routes: () => [],
    selectedRouteIndex: 0,
    departureStation: null,
    destinationStation: null,
    intermediateStations: () => [],
    height: '100%',
  },
)

const mapRef = ref<InstanceType<typeof LMap> | null>(null)
const mapReady = ref(false)

const center = ref<[number, number]>([46.603354, 1.888334])
const zoom = ref(6)

const tileUrl = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
const tileAttribution =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CartoDB</a>'

const ROUTE_COLORS = ['#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']

function onMapReady() {
  mapReady.value = true
}

function fitToSegments(segments: RouteSegment[]) {
  if (!mapRef.value) return

  const allPoints: [number, number][] = []
  for (const segment of segments) {
    for (const point of segment.geometry) {
      allPoints.push(point)
    }
  }
  if (allPoints.length === 0) return

  try {
    const bounds = L.latLngBounds(allPoints.map((p) => L.latLng(p[0], p[1])))
    if (!bounds.isValid()) return

    const leafletMap = (mapRef.value as unknown as { leafletObject: L.Map }).leafletObject
    if (leafletMap) {
      leafletMap.fitBounds(bounds, { padding: [50, 50] })
    }
  } catch {
    // Map not ready yet
  }
}

// Fit to primary route or first multi-route
watch(
  [() => props.route, () => props.routes, mapReady],
  ([segments, multiRoutes]) => {
    if (!mapReady.value) return
    if (segments && segments.length > 0) {
      fitToSegments(segments)
    } else if (multiRoutes && multiRoutes.length > 0) {
      fitToSegments(multiRoutes[0].segments)
    }
  },
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
      @ready="onMapReady"
    >
      <LTileLayer :url="tileUrl" :attribution="tileAttribution" />

      <!-- Multi-route mode (MOA*) : draw non-selected routes first (behind) -->
      <template v-if="routes.length > 1">
        <RouteLayer
          v-for="(r, i) in routes"
          :key="'route-' + i"
          :segments="r.segments"
          :color="ROUTE_COLORS[i % ROUTE_COLORS.length]"
          :opacity="i === selectedRouteIndex ? 0.9 : 0.25"
          :label="r.label"
        />
      </template>

      <!-- Single route mode -->
      <RouteLayer v-else-if="route" :segments="route" />

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

import { ref, computed } from 'vue'
import type { ResolveResponse, RouteSegment, Station } from '../api/types'

export function useMapRoute() {
  const routeSegments = ref<RouteSegment[]>([])
  const departureStation = ref<Station | null>(null)
  const destinationStation = ref<Station | null>(null)
  const intermediateStations = ref<Station[]>([])

  const hasRoute = computed(() => routeSegments.value.length > 0)

  function updateFromResponse(response: ResolveResponse): void {
    routeSegments.value = []
    departureStation.value = null
    destinationStation.value = null
    intermediateStations.value = []

    if (!response.pathfinding?.found || !response.pathfinding.route_details) return

    routeSegments.value = response.pathfinding.route_details

    const details = response.pathfinding.route_details
    if (details.length === 0) return

    const first = details[0]!
    const last = details[details.length - 1]!

    if (first.geometry.length > 0) {
      const pt = first.geometry[0]!
      departureStation.value = {
        name: first.from_station,
        uic: first.from_uic,
        lat: pt[0],
        lon: pt[1],
      }
    }

    if (last.geometry.length > 0) {
      const lastPoint = last.geometry[last.geometry.length - 1]!
      destinationStation.value = {
        name: last.to_station,
        uic: last.to_uic,
        lat: lastPoint[0],
        lon: lastPoint[1],
      }
    }

    // Only show markers for transfer stations (line changes)
    for (let i = 0; i < details.length - 1; i++) {
      const seg = details[i]!
      const nextSeg = details[i + 1]!
      // Transfer = different line, or walk segment
      const isTransfer = seg.line !== nextSeg.line || seg.type === 'WALK' || nextSeg.type === 'WALK'
      if (isTransfer && seg.geometry.length > 0) {
        const point = seg.geometry[seg.geometry.length - 1]!
        intermediateStations.value.push({
          name: seg.to_station,
          uic: seg.to_uic,
          lat: point[0],
          lon: point[1],
        })
      }
    }
  }

  function clearRoute(): void {
    routeSegments.value = []
    departureStation.value = null
    destinationStation.value = null
    intermediateStations.value = []
  }

  return {
    routeSegments,
    departureStation,
    destinationStation,
    intermediateStations,
    hasRoute,
    updateFromResponse,
    clearRoute,
  }
}

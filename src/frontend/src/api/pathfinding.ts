import { apiGet, apiPost } from './client'
import type { Station, RouteRequest, PathfindingResult } from './types'

export function searchStations(query: string, limit = 10) {
  return apiGet<{ stations: Station[] }>(
    `/api/pathfinding/stations?q=${encodeURIComponent(query)}&limit=${limit}`,
  )
}

export function findRoute(req: RouteRequest) {
  return apiPost<PathfindingResult>('/api/pathfinding/route', req)
}

// Enums
export type Intent = 'TRIP' | 'NOT_TRIP' | 'UNKNOWN'
export type Language = 'FRENCH' | 'ENGLISH' | 'UNKNOWN'
export type SegmentType = 'TRAIN' | 'WALK'

// Health
export interface HealthResponse {
  status: string
  version: string
  models_loaded: string[]
  stations_count: number
  graph_nodes: number
  graph_edges: number
  device: string
  gpu_name?: string
}

// NLP sub-types
export interface LanguageResult {
  detected: Language
  confidence: number
  model: string
}

export interface IntentResult {
  value: Intent
  confidence: number
  model: string
}

export interface EntityMatch {
  raw: string
  matched: string | null
  confidence?: number
}

export interface EntitiesResult {
  departure: EntityMatch | null
  destination: EntityMatch | null
  intermediates: EntityMatch[]
  model: string
  fuzzy_enabled: boolean
}

export interface NlpResult {
  language: LanguageResult
  intent: IntentResult
  entities: EntitiesResult
  processed_text: string
  latency_ms: number
}

// Pathfinding
export interface RouteSegment {
  from_station: string
  from_uic: string
  to_station: string
  to_uic: string
  line: string
  type: SegmentType
  geometry: [number, number][]
}

export interface PathfindingResult {
  found: boolean
  route: string[] | null
  route_details: RouteSegment[] | null
  total_stops: number
  transfers: number
  error: string | null
}

// Resolve (full pipeline)
export interface ResolveRequest {
  text: string
  intent_model?: string
  entity_model?: string
  use_fuzzy?: boolean
}

export interface ResolveResponse {
  nlp: NlpResult
  pathfinding: PathfindingResult | null
}

// NLP individual endpoints
export interface NlpLanguageItem {
  model: string
  detected: Language
  confidence: number
  latency_ms: number
}

export interface NlpIntentItem {
  model: string
  intent: Intent
  confidence: number
  latency_ms: number
}

export interface NlpEntityItem {
  model: string
  fuzzy: boolean
  departure: EntityMatch | null
  destination: EntityMatch | null
  intermediates: EntityMatch[]
  latency_ms: number
}

// Pathfinding
export interface Station {
  name: string
  uic: string
  lat: number
  lon: number
}

export interface RouteRequest {
  departure: string
  destination: string
  intermediates?: string[]
}

// Evaluation
export interface EvalReport {
  id: string
  date: string
  eval_type: string
  dataset: string
  samples: number
}

export interface ClassMetrics {
  precision: number
  recall: number
  f1: number
  support: number
}

export interface ConfusionMatrix {
  labels: string[]
  matrix: number[][]
}

export interface ModelEvalResult {
  accuracy: number
  per_class: Record<string, ClassMetrics>
  per_language?: Record<string, number>
  avg_latency_ms: number
  confusion_matrix?: ConfusionMatrix
}

export interface EvalReportDetail {
  id: string
  date: string
  config: {
    eval_type: string
    dataset: string
    samples: number
    device: string
    preprocess: boolean
  }
  language?: Record<string, ModelEvalResult>
  intent?: Record<string, ModelEvalResult>
  entity?: Record<string, ModelEvalResult>
  entity_fuzzy?: Record<string, ModelEvalResult>
}

export interface EvalRunRequest {
  eval_type: string
  intent_models?: string[]
  entity_models?: string[]
  language_models?: string[]
  use_fuzzy?: boolean
  preprocess?: boolean
  device?: string
  dataset?: string
}

export interface EvalRunResponse {
  task_id: string
  status: string
  message: string
}

export interface EvalStatus {
  task_id: string
  status: 'started' | 'running' | 'completed' | 'failed'
  progress: {
    current_step?: string
    steps_completed?: number
    steps_total?: number
    percent: number
    elapsed_seconds: number
  }
  report_id?: string
}

// Speech
export interface TranscriptionResponse {
  text: string
  language: string
  confidence: number
  duration_seconds: number
  latency_ms: number
}

// Monitoring
export interface RequestMetric {
  timestamp: string
  input_text: string
  model_name: string
  duration_s: number
  cpu_percent_avg: number | null
  ram_peak_mb: number | null
  gpu_peak_mb: number | null
  carbon_kg: number | null
}

export interface MonitoringMetricsResponse {
  requests: RequestMetric[]
  total_requests: number
  avg_latency_ms: number
}

export interface ResourceSnapshot {
  cpu_percent: number
  ram_used_mb: number
  ram_total_mb: number
  ram_percent: number
  gpu_used_mb: number | null
  gpu_total_mb: number | null
  gpu_percent: number | null
}

export interface CarbonSummary {
  total_emissions_kg: number
  total_energy_kwh: number
  total_requests: number
  country: string
}

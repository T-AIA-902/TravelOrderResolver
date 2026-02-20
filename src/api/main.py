"""Main FastAPI application with lifespan and CORS."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import evaluation, health, monitoring, nlp, pathfinding, resolve, speech
from src.data import StationDatabase
from src.evaluation.model_factory import (
    create_entity_extractors,
    create_fuzzy_post_processor,
    create_intent_classifiers,
    create_language_detectors,
)
from src.monitoring.metrics_logger import MetricsLogger
from src.pathfinding.graph import TrainGraph
from src.speech.whisper_model import WhisperModel
from src.utils.device import get_device_info, get_torch_device

DeviceType = str  # "auto" | "cuda" | "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[arg-type]
    device: DeviceType = get_torch_device("auto")
    app.state.device = device
    app.state.device_info = get_device_info()
    app.state.graph = TrainGraph()
    app.state.station_db = StationDatabase()
    app.state.station_db.load()
    app.state.metrics_logger = MetricsLogger(output_dir="reports/metrics")
    app.state.language_detectors = create_language_detectors(["all"])
    app.state.intent_classifiers = create_intent_classifiers(
        ["all"], device=device  # type: ignore[arg-type]
    )
    app.state.entity_extractors = create_entity_extractors(
        ["all"], device=device  # type: ignore[arg-type]
    )
    app.state.fuzzy_post = create_fuzzy_post_processor()
    app.state.whisper = WhisperModel(model_name="base", device="auto")
    app.state.eval_tasks = {}
    yield
    if app.state.whisper.is_loaded:
        app.state.whisper.unload()


app = FastAPI(title="Travel Order Resolver API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(resolve.router, prefix="/api")
app.include_router(nlp.router, prefix="/api")
app.include_router(pathfinding.router, prefix="/api")
app.include_router(speech.router, prefix="/api")
app.include_router(evaluation.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")

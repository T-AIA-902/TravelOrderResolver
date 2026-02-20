"""Health check endpoint."""

from fastapi import APIRouter, Depends

from src.api.dependencies import (
    get_device,
    get_device_info_data,
    get_entity_extractors,
    get_graph,
    get_intent_classifiers,
)

router = APIRouter()


@router.get("/health")
def health_check(
    graph=Depends(get_graph),
    intent_classifiers=Depends(get_intent_classifiers),
    entity_extractors=Depends(get_entity_extractors),
    device=Depends(get_device),
    device_info=Depends(get_device_info_data),
):
    model_names = set()
    for name, _ in intent_classifiers:
        model_names.add(name.lower())
    for name, _ in entity_extractors:
        model_names.add(name.lower())

    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": sorted(model_names),
        "stations_count": graph.graph.number_of_nodes(),
        "graph_nodes": graph.graph.number_of_nodes(),
        "graph_edges": graph.graph.number_of_edges(),
        "device": device,
        "gpu_name": device_info.get("cuda_device_name"),
    }

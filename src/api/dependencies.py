"""FastAPI dependency injection functions."""

from fastapi import Request


def get_graph(request: Request):
    return request.app.state.graph


def get_station_db(request: Request):
    return request.app.state.station_db


def get_metrics_logger(request: Request):
    return request.app.state.metrics_logger


def get_whisper(request: Request):
    return request.app.state.whisper


def get_language_detectors(request: Request):
    return request.app.state.language_detectors


def get_intent_classifiers(request: Request):
    return request.app.state.intent_classifiers


def get_entity_extractors(request: Request):
    return request.app.state.entity_extractors


def get_fuzzy_post(request: Request):
    return request.app.state.fuzzy_post


def get_eval_tasks(request: Request):
    return request.app.state.eval_tasks


def get_device(request: Request):
    return request.app.state.device


def get_device_info_data(request: Request):
    return request.app.state.device_info

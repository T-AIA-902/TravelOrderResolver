"""Evaluation endpoints: run NLP evaluations and browse reports."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_eval_tasks

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

REPORTS_DIR = Path("src/evaluation/reports")


# ── Request schemas ──────────────────────────────────────────────────────────


class EvalRunRequest(BaseModel):
    eval_type: str
    intent_models: list[str] | None = None
    entity_models: list[str] | None = None
    use_fuzzy: bool = True
    device: str = "cpu"
    max_samples: int | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/reports")
def list_reports():
    """List available evaluation reports, most recent first."""
    reports: list[dict] = []
    if not REPORTS_DIR.exists():
        return {"reports": reports}

    for path in sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.name, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            parts = path.stem.split("_")
            date_str = parts[1] if len(parts) >= 2 else "unknown"
            reports.append(
                {
                    "id": path.stem,
                    "date": date_str,
                    "eval_type": data.get("eval_type", "all"),
                    "dataset": data.get("dataset", "test_set"),
                    "samples": data.get("samples", 0),
                }
            )
        except Exception:
            continue

    return {"reports": reports}


@router.get("/reports/{report_id}")
def get_report(report_id: str):
    """Return the full JSON content of a single evaluation report."""
    path = REPORTS_DIR / f"{report_id}.json"
    if not path.exists():
        raise HTTPException(404, detail=f"Rapport non trouvé : {report_id}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/run")
def run_evaluation(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
    eval_tasks: dict = Depends(get_eval_tasks),
):
    """Launch an evaluation in the background and return a task id."""
    task_id = str(uuid.uuid4())
    eval_tasks[task_id] = {
        "status": "running",
        "progress": {
            "percent": 0,
            "elapsed_seconds": 0,
            "current_step": "Initialisation...",
        },
        "report_id": None,
    }

    background_tasks.add_task(_run_evaluation, task_id, eval_tasks, request)

    return {
        "task_id": task_id,
        "status": "started",
        "message": f"Évaluation '{request.eval_type}' lancée",
    }


@router.get("/status/{task_id}")
def get_status(task_id: str, eval_tasks: dict = Depends(get_eval_tasks)):
    """Return the current status and progress of an evaluation task."""
    task = eval_tasks.get(task_id)
    if task is None:
        raise HTTPException(404, detail=f"Tâche non trouvée : {task_id}")
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "report_id": task["report_id"],
    }


# ── Background runner ────────────────────────────────────────────────────────


def _run_evaluation(task_id: str, eval_tasks: dict, request: EvalRunRequest) -> None:
    """Execute the full evaluation pipeline in a background thread."""
    from datetime import datetime

    from src.evaluation.data_loader import load_dataset
    from src.evaluation.evaluators import (
        evaluate_combined,
        evaluate_entity_extractors,
        evaluate_intent_classifiers,
        evaluate_language_detectors,
    )
    from src.evaluation.model_factory import (
        create_entity_extractors,
        create_fuzzy_post_processor,
        create_intent_classifiers,
        create_language_detectors,
    )
    from src.evaluation.reporting import export_results_json

    start_time = time.time()

    completed_items: list[dict] = []

    def _update(step: str, step_num: int, total_steps: int, sub: int = 0, sub_total: int = 0):
        """Update task progress with global step tracking."""
        base = int((step_num / total_steps) * 100)
        step_size = 100 / total_steps
        sub_pct = int((sub / sub_total) * step_size) if sub_total > 0 else 0
        percent = min(99, base + sub_pct)
        eval_tasks[task_id]["progress"] = {
            "percent": percent,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "current_step": f"[{step_num + 1}/{total_steps}] {step}" + (
                f" ({sub}/{sub_total})" if sub_total > 0 else ""
            ),
            "steps_completed": step_num,
            "steps_total": total_steps,
            "completed": completed_items,
        }

    def _log_result(category: str, model: str, metric: str, value: float):
        """Log a completed evaluation result."""
        completed_items.append({
            "category": category,
            "model": model,
            "metric": metric,
            "value": round(value, 4),
        })

    try:
        # Load dataset
        _update("Chargement du dataset...", 0, 1)
        data = load_dataset("datasets/augmented/test.csv")

        # Limit dataset size if requested
        if request.max_samples and len(data) > request.max_samples:
            import random
            random.seed(42)
            data = random.sample(data, request.max_samples)

        # Resolve model lists
        eval_type = request.eval_type
        intent_models = request.intent_models or ["all"]
        entity_models = request.entity_models or ["all"]
        device = request.device
        use_fuzzy = request.use_fuzzy

        # Determine which evaluations to run
        need_intent = eval_type in ("intent", "combined", "combined_fuzzy", "all")
        need_entity = eval_type in ("entity", "entity_fuzzy", "combined", "combined_fuzzy", "all")
        need_language = eval_type in ("language", "all")
        need_fuzzy = use_fuzzy and need_entity

        # Build the list of steps to compute total_steps for progress
        steps: list[str] = ["Chargement modèles"]
        if need_language:
            steps.append("Langue")
        if need_intent:
            steps.append("Intent")
        if eval_type in ("entity", "all"):
            steps.append("Entités")
        if eval_type in ("entity_fuzzy", "all"):
            steps.append("Entités + Fuzzy")
        if eval_type in ("combined", "all"):
            steps.append("Combiné")
        if eval_type in ("combined_fuzzy", "all"):
            steps.append("Combiné + Fuzzy")
        steps.append("Export")
        total_steps = len(steps)
        step_idx = 0

        # Create models
        _update("Chargement des modèles...", step_idx, total_steps)
        classifiers = (
            create_intent_classifiers(intent_models, device=device)  # type: ignore[arg-type]
            if need_intent
            else []
        )
        extractors = (
            create_entity_extractors(entity_models, device=device)  # type: ignore[arg-type]
            if need_entity
            else []
        )
        fuzzy_post = create_fuzzy_post_processor() if need_fuzzy else None
        language_detectors = create_language_detectors(["all"]) if need_language else []
        step_idx += 1

        # Run evaluations
        from src.evaluation.metrics import (
            CombinedResults,
            EntityResults,
            IntentResults,
            LanguageResults,
        )

        intent_results: dict[str, IntentResults] = {}
        entity_results: dict[str, EntityResults] = {}
        entity_fuzzy_results: dict[str, EntityResults] = {}
        combined_results: list[CombinedResults] = []
        combined_fuzzy_results: list[CombinedResults] = []
        language_results: dict[str, LanguageResults] = {}

        def make_progress_cb(step_name: str, si: int):
            """Create a progress callback bound to a specific step."""
            def cb(current: int, total: int):
                _update(step_name, si, total_steps, current, total)
            return cb

        if need_language:
            _update("Évaluation langue...", step_idx, total_steps)
            language_results = evaluate_language_detectors(
                language_detectors, data,
                progress_callback=make_progress_cb("Évaluation langue", step_idx),
            )
            for name, res in language_results.items():
                _log_result("Langue", name, "accuracy", res.correct / res.total if res.total else 0)
            step_idx += 1

        if need_intent:
            _update("Évaluation intent...", step_idx, total_steps)
            intent_results = evaluate_intent_classifiers(
                classifiers, data,
                progress_callback=make_progress_cb("Évaluation intent", step_idx),
            )
            for name, res in intent_results.items():
                _log_result("Intent", name, "accuracy", res.correct / res.total if res.total else 0)
            step_idx += 1

        if eval_type in ("entity", "all"):
            _update("Évaluation entités...", step_idx, total_steps)
            entity_results = evaluate_entity_extractors(
                extractors, data,
                progress_callback=make_progress_cb("Évaluation entités", step_idx),
            )
            for name, res in entity_results.items():
                _log_result("Entity", name, "accuracy", res.correct / res.total if res.total else 0)
            step_idx += 1

        if eval_type in ("entity_fuzzy", "all"):
            _update("Évaluation entités + fuzzy...", step_idx, total_steps)
            entity_fuzzy_results = evaluate_entity_extractors(
                extractors, data,
                fuzzy_post=fuzzy_post,
                normalize_fuzzy=True,
                progress_callback=make_progress_cb("Évaluation entités + fuzzy", step_idx),
            )
            for name, res in entity_fuzzy_results.items():
                _log_result("Entity+Fuzzy", name, "accuracy", res.correct / res.total if res.total else 0)
            step_idx += 1

        if eval_type in ("combined", "all"):
            _update("Évaluation combinée...", step_idx, total_steps)
            combined_results = evaluate_combined(
                classifiers, extractors, data,
                progress_callback=make_progress_cb("Évaluation combinée", step_idx),
            )
            step_idx += 1

        if eval_type in ("combined_fuzzy", "all"):
            _update("Évaluation combinée + fuzzy...", step_idx, total_steps)
            combined_fuzzy_results = evaluate_combined(
                classifiers, extractors, data,
                fuzzy_post=fuzzy_post,
                normalize_fuzzy=True,
                progress_callback=make_progress_cb("Évaluation combinée + fuzzy", step_idx),
            )
            step_idx += 1

        # Export results
        _update("Export des résultats...", step_idx, total_steps)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = REPORTS_DIR / f"evaluation_{timestamp}.json"

        export_results_json(
            intent_results,
            entity_results,
            entity_fuzzy_results,
            combined_results,
            combined_fuzzy_results,
            language_results,
            str(output_path),
        )

        eval_tasks[task_id]["status"] = "completed"
        eval_tasks[task_id]["report_id"] = output_path.stem
        eval_tasks[task_id]["progress"] = {
            "percent": 100,
            "elapsed_seconds": round(time.time() - start_time, 1),
            "current_step": "Terminé",
        }

    except Exception as exc:
        eval_tasks[task_id]["status"] = "failed"
        eval_tasks[task_id]["progress"]["current_step"] = f"Erreur : {exc}"

"""Integrated Gradients through the hetero-GNN patient node (Level 1, GNN half).

The GNN's patient node input *is* the 12 scaled features
(``build_hetero_clinical_graph`` sets ``data["patient"].x`` to the scaled
matrix), which is what makes attribution clean here: "which biomarker moved the
GNN's vote" reduces to a gradient with respect to that node's features.

Two properties of the deployed model shape this implementation
-------------------------------------------------------------
**Batch size must be 1.** The 12 biomarker nodes are shared across every patient
in a batch, and ``SAGEConv`` aggregates ``patient -> biomarker``. A patient's
output therefore depends on the other rows present. ``model_client.grade_record``
always calls ``grader.grade`` with a single row, so deployed inference is
batch-1 -- and the explanation has to match it. Explaining a batch would answer
a question nobody asked.

**Gradients need a fresh forward.** ``AnemiaGrader.predict_gnn`` is wrapped in
``@torch.inference_mode()``, which disables autograd. This module calls the GNN
directly instead of reusing that method.

Baseline choice
---------------
The baseline is the zero vector in *scaled* space, i.e. the training mean. So a
positive attribution reads as "this patient's value sits above the population
average in a way that pushed the vote toward class k". That is the interpretable
framing; a zero baseline in raw units would mean "MCV of 0", which is clinical
nonsense.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from ..grader_model import gnn as gnn_module
from ..grader_model.features import FEATURE_NAMES
from .stage_attribution import ATTRIBUTION_CLASSES

DEFAULT_STEPS = 32


@dataclass
class GnnAttribution:
    """Integrated-gradients attribution for one patient and one class."""

    class_index: int
    class_name: str
    #: feature name -> attribution in scaled units (log-odds-ish, pre-softmax)
    scaled_contributions: Dict[str, float] = field(default_factory=dict)
    #: same, converted to clinical units per feature (divide by scaler.scale_)
    clinical_contributions: Dict[str, float] = field(default_factory=dict)
    steps: int = DEFAULT_STEPS
    baseline: str = "scaled-zero (training mean)"


def _knowledge_edges_present() -> bool:
    """``build_hetero_clinical_graph`` reads a module-global set by ``load_gnn``.

    Importing and calling it without ever having loaded the model yields a graph
    with *no* clinical-knowledge edges -- silently a different model. Catch that
    rather than explaining the wrong graph.
    """
    return bool(getattr(gnn_module, "_knowledge_edges", None))


def _forward_logits(
    model: Any,
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[Any, torch.Tensor],
) -> torch.Tensor:
    """Forward pass returning raw logits (pre-softmax), gradient-enabled."""
    return model(x_dict, edge_index_dict)


def _as_single_row_frame(scaled_row: Any) -> pd.DataFrame:
    """Coerce input into exactly one row of the 12 scaled features, or raise."""
    if isinstance(scaled_row, pd.DataFrame):
        frame = scaled_row[FEATURE_NAMES]
    else:
        arr = np.asarray(scaled_row, dtype=float).reshape(1, -1)
        if arr.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} scaled features, got {arr.shape[1]}."
            )
        frame = pd.DataFrame(arr, columns=FEATURE_NAMES)

    if len(frame) != 1:
        raise ValueError(
            "explain_gnn expects exactly one patient. The shared biomarker nodes "
            "make the GNN's output batch-dependent, so a batch explanation would "
            "not describe deployed (batch-1) inference."
        )
    return frame


def _logits_at(gnn: Any, frame: pd.DataFrame) -> torch.Tensor:
    """Raw logits for a single-row frame, no gradients."""
    graph = gnn_module.build_hetero_clinical_graph(frame)
    x_dict = {nt: graph[nt].x for nt in graph.node_types}
    edge_index_dict = {et: graph[et].edge_index for et in graph.edge_types}
    with torch.no_grad():
        return gnn(x_dict, edge_index_dict)


def explain_gnn(
    gnn: Any,
    scaled_row: Any,
    scaler: Optional[Any] = None,
    steps: int = DEFAULT_STEPS,
    class_indices: Optional[List[int]] = None,
) -> List[GnnAttribution]:
    """Integrated Gradients for a single patient through the hetero-GNN.

    ``scaled_row`` is one row of the **scaled** 12-feature matrix (a DataFrame
    row, a 1-D array, or a 2-D single-row array) -- the GNN is trained on scaled
    features while the cascade is trained on imputed ones. Passing imputed
    values here would be the mirror-image mistake of ``explain_xgb``.

    ``scaler`` is optional and only used to convert attributions into clinical
    units; pass ``AnemiaGrader().scaler`` to get that column populated.
    """
    if not _knowledge_edges_present():
        raise RuntimeError(
            "GNN knowledge edges are unset. Call grader_model.gnn.load_gnn(...) "
            "(normally via AnemiaGrader()) before explaining, otherwise the graph "
            "has no clinical-knowledge edges and is not the deployed model."
        )

    frame = _as_single_row_frame(scaled_row)

    graph = gnn_module.build_hetero_clinical_graph(frame)
    x_dict = {nt: graph[nt].x for nt in graph.node_types}
    edge_index_dict = {et: graph[et].edge_index for et in graph.edge_types}

    x_patient = x_dict["patient"].detach().clone().requires_grad_(False)
    baseline = torch.zeros_like(x_patient)

    was_training = gnn.training
    gnn.eval()
    try:
        alphas = torch.linspace(0.0, 1.0, steps).view(-1, 1)
        targets = class_indices if class_indices is not None else list(range(4))

        accumulated: Dict[int, torch.Tensor] = {
            k: torch.zeros_like(x_patient) for k in targets
        }
        for alpha in alphas:
            point = (baseline + alpha * (x_patient - baseline)).detach().clone()
            point.requires_grad_(True)
            step_x = dict(x_dict)
            step_x["patient"] = point
            logits = _forward_logits(gnn, step_x, edge_index_dict)
            for k in targets:
                grad = torch.autograd.grad(
                    logits[0, k], point, retain_graph=True, allow_unused=True
                )[0]
                if grad is not None:
                    accumulated[k] = accumulated[k] + grad
    finally:
        if was_training:  # pragma: no cover - model is loaded in eval mode
            gnn.train()

    delta = (x_patient - baseline).detach()
    results: List[GnnAttribution] = []
    for k in targets:
        attribution = delta * (accumulated[k] / float(steps))
        flat = attribution.reshape(-1).detach().numpy()

        scaled_contrib = {name: float(flat[j]) for j, name in enumerate(FEATURE_NAMES)}
        clinical_contrib: Dict[str, float] = {}
        if scaler is not None and hasattr(scaler, "scale_"):
            # StandardScaler: scaled = (x - mean) / scale, so d/dx = 1 / scale.
            inverse_scale = 1.0 / np.asarray(scaler.scale_, dtype=float)
            clinical_contrib = {
                name: float(flat[j] * inverse_scale[j]) for j, name in enumerate(FEATURE_NAMES)
            }

        results.append(
            GnnAttribution(
                class_index=k,
                class_name=ATTRIBUTION_CLASSES[k],
                scaled_contributions=scaled_contrib,
                clinical_contributions=clinical_contrib,
                steps=steps,
            )
        )
    return results


def completeness_gap(
    gnn: Any,
    scaled_row: Any,
    class_index: int,
    steps: int = DEFAULT_STEPS,
) -> float:
    """``sum(attributions) - (f(x) - f(baseline))`` for one class.

    Integrated Gradients is *complete*: the attributions sum to the change in
    the explained function between the input and the baseline. That identity
    holds exactly only in the limit of infinitely many interpolation steps, so a
    finite ``steps`` leaves a residual. This exposes it, letting tests bound the
    error rather than trusting the method.
    """
    attribution = explain_gnn(
        gnn, scaled_row, steps=steps, class_indices=[class_index]
    )[0]
    total = float(sum(attribution.scaled_contributions.values()))

    frame = _as_single_row_frame(scaled_row)
    zero_frame = pd.DataFrame(np.zeros((1, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    delta = (
        _logits_at(gnn, frame)[0, class_index]
        - _logits_at(gnn, zero_frame)[0, class_index]
    )
    return total - float(delta)

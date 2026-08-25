"""Observation builders for neural MARL policies."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from asiot.datatypes import SimulationConfig, Task
from asiot.node import AgenticNode
from asiot.social_cognition import clip01

OBSERVATION_DIM = 23


def build_candidate_observation(
    requester_node: AgenticNode,
    target_node: AgenticNode,
    task: Task,
    score_components: dict[str, float | int | str],
    graph: Any,
    config: SimulationConfig,
    include_social_features: bool = True,
) -> list[float]:
    """Build a fixed normalized observation for one requester-target candidate.

    The observation combines requester resources, target resources, task
    requirements, load, social-cognitive scores, privacy state, link quality,
    interaction probability, expected success, and total utility.
    """
    requester_energy, requester_bandwidth, requester_compute = (
        requester_node.state.resources.as_vector()
    )
    target_energy, target_bandwidth, target_compute = target_node.state.resources.as_vector()
    load_factor = _load_factor(config.load_level)

    local_trust = _score(score_components, "local_trust", 0.5)
    reputation = _score(score_components, "reputation", 0.5)
    effective_trust = _score(score_components, "effective_trust", 0.5)
    preference = _score(score_components, "preference", 0.5)
    reciprocity = _score(score_components, "reciprocity", 0.5)
    privacy_risk = _score(score_components, "privacy_risk", 0.5)
    privacy_allowed = _score(score_components, "privacy_allowed", 1.0)

    if not include_social_features:
        local_trust = 0.5
        reputation = 0.5
        effective_trust = 0.5
        preference = 0.5
        reciprocity = 0.5
        privacy_risk = 0.5
        privacy_allowed = 1.0

    values = [
        requester_energy,
        requester_bandwidth,
        requester_compute,
        target_energy,
        target_bandwidth,
        target_compute,
        task.required_compute,
        task.required_bandwidth,
        task.complexity,
        task.data_sensitivity,
        load_factor,
        local_trust,
        reputation,
        effective_trust,
        preference,
        reciprocity,
        privacy_risk,
        privacy_allowed,
        _score(score_components, "qos", 0.0),
        _score(score_components, "distance_cost", 1.0),
        _score(score_components, "interaction_probability", 0.0),
        _score(score_components, "expected_success", 0.0),
        _score(score_components, "total_utility", 0.0),
    ]
    if len(values) != OBSERVATION_DIM:
        raise ValueError(f"candidate observation must have {OBSERVATION_DIM} values.")
    return [_finite_unit(value) for value in values]


def build_top_k_observation(
    requester_node: AgenticNode,
    task: Task,
    neighbor_scores: dict[int, dict[str, float | int | str]],
    nodes: dict[int, AgenticNode],
    graph: Any,
    config: SimulationConfig,
    top_k_candidates: int = 8,
    include_social_features: bool = True,
    sort_field: str = "total_utility",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return flattened top-K observations, an action mask, and candidate IDs."""
    ranked = sorted(
        neighbor_scores.items(),
        key=lambda item: (
            float(item[1].get(sort_field, 0.0)),
            float(item[1].get("interaction_probability", 0.0)),
        ),
        reverse=True,
    )[:top_k_candidates]
    candidate_ids = [neighbor_id for neighbor_id, _ in ranked]
    candidate_vectors = []
    for neighbor_id, score in ranked:
        candidate_vectors.extend(
            build_candidate_observation(
                requester_node,
                nodes[neighbor_id],
                task,
                score,
                graph,
                config,
                include_social_features=include_social_features,
            )
        )

    missing = top_k_candidates - len(ranked)
    if missing > 0:
        candidate_vectors.extend([0.0] * missing * OBSERVATION_DIM)

    action_mask = np.zeros(top_k_candidates + 1, dtype=np.float32)
    for index, (_, score) in enumerate(ranked):
        if int(score.get("privacy_allowed", 1)) == 1 and float(
            score.get("interaction_probability", 0.0)
        ) > 0.0:
            action_mask[index] = 1.0
    action_mask[top_k_candidates] = 1.0
    return (
        np.asarray(candidate_vectors, dtype=np.float32),
        action_mask,
        candidate_ids,
    )


def _score(score_components: dict[str, float | int | str], key: str, default: float) -> float:
    return _finite_unit(float(score_components.get(key, default)))


def _finite_unit(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("observation values must be finite.")
    return clip01(value)


def _load_factor(load_level: str) -> float:
    return {
        "low": 0.00,
        "medium": 0.33,
        "high": 0.66,
        "extreme": 1.00,
    }.get(load_level, 0.33)

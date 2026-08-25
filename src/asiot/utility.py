"""Utility-driven proposed model for the Agentic SIoT framework."""

from __future__ import annotations

from dataclasses import dataclass

import math
from collections.abc import Mapping

from asiot.datatypes import SimulationConfig, Task
from asiot.social_cognition import clip01

UTILITY_WEIGHT_KEYS = (
    "system",
    "social",
    "resource",
    "privacy",
    "incentive",
)



@dataclass(frozen=True)
class SuccessWeights:
    """Additive weights of the expected-success model (Eq. 27).

    Grouped into one object so the coefficients can be swept without changing
    call signatures. Defaults reproduce the previously hardcoded constants
    exactly. Model identity is deliberately absent: these weights are the same
    for every policy (see tests/test_model_identity_guard.py).
    """

    base: float = 0.18
    probability: float = 0.25
    link_quality: float = 0.15
    requester_resource: float = 0.10
    target_resource: float = 0.15
    trust: float = 0.15
    load: float = 0.10
    complexity: float = 0.10


DEFAULT_SUCCESS_WEIGHTS = SuccessWeights()


def success_weights_from_config(config: SimulationConfig) -> SuccessWeights:
    """Read the Eq. 27 weights from configuration."""
    return SuccessWeights(
        base=config.success_base,
        probability=config.success_w_probability,
        link_quality=config.success_w_link_quality,
        requester_resource=config.success_w_requester_resource,
        target_resource=config.success_w_target_resource,
        trust=config.success_w_trust,
        load=config.success_w_load,
        complexity=config.success_w_complexity,
    )


def validate_utility_weights(weights: Mapping[str, float]) -> None:
    """Validate total utility weights for the proposed A-SIoT model."""
    if set(weights) != set(UTILITY_WEIGHT_KEYS):
        raise ValueError(f"weights must contain keys {UTILITY_WEIGHT_KEYS}.")
    for key, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"weight {key!r} must be finite and nonnegative.")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("utility weights must sum to approximately 1.0.")


def system_utility(
    expected_success: float,
    delay_norm: float,
    alpha_success: float = 0.7,
    alpha_delay: float = 0.3,
) -> float:
    """Compute system utility from expected success and normalized delay."""
    _validate_unit("expected_success", expected_success)
    _validate_unit("delay_norm", delay_norm)
    _validate_nonnegative("alpha_success", alpha_success)
    _validate_nonnegative("alpha_delay", alpha_delay)
    if not math.isclose(alpha_success + alpha_delay, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("system utility weights must sum to approximately 1.0.")
    return clip01(alpha_success * expected_success + alpha_delay * (1.0 - delay_norm))


def social_utility(
    effective_trust: float,
    preference: float,
    reciprocity: float,
    beta_trust: float = 0.4,
    beta_preference: float = 0.3,
    beta_reciprocity: float = 0.3,
) -> float:
    """Compute social utility from trust, preference, and reciprocity."""
    _validate_unit("effective_trust", effective_trust)
    _validate_unit("preference", preference)
    _validate_unit("reciprocity", reciprocity)
    _validate_sum_one((beta_trust, beta_preference, beta_reciprocity))
    return clip01(
        beta_trust * effective_trust
        + beta_preference * preference
        + beta_reciprocity * reciprocity
    )


def resource_utility(
    energy_cost: float,
    bandwidth_cost: float,
    compute_cost: float,
    gamma_energy: float = 0.4,
    gamma_bandwidth: float = 0.3,
    gamma_compute: float = 0.3,
) -> float:
    """Compute resource utility as one minus weighted resource cost."""
    _validate_unit("energy_cost", energy_cost)
    _validate_unit("bandwidth_cost", bandwidth_cost)
    _validate_unit("compute_cost", compute_cost)
    _validate_sum_one((gamma_energy, gamma_bandwidth, gamma_compute))
    cost = gamma_energy * energy_cost + gamma_bandwidth * bandwidth_cost + gamma_compute * compute_cost
    return clip01(1.0 - cost)


def privacy_utility(privacy_risk: float, delta_privacy: float = 1.0) -> float:
    """Compute privacy utility as one minus weighted privacy risk."""
    _validate_unit("privacy_risk", privacy_risk)
    _validate_nonnegative("delta_privacy", delta_privacy)
    return clip01(1.0 - delta_privacy * privacy_risk)


def fairness_utility(
    target_resource_score: float,
    target_load: float,
    eta_resource: float = 0.5,
    eta_load: float = 0.5,
) -> float:
    """Compute fairness/workload-balance utility.

    Higher values prefer capable but less overloaded partners.
    """
    _validate_unit("target_resource_score", target_resource_score)
    _validate_unit("target_load", target_load)
    _validate_sum_one((eta_resource, eta_load))
    return clip01(eta_resource * target_resource_score + eta_load * (1.0 - target_load))


def incentive_utility(
    cooperation_history: float,
    efficiency_score: float,
    effective_trust: float,
    zeta_coop: float = 0.4,
    zeta_eff: float = 0.3,
    zeta_trust: float = 0.3,
) -> float:
    """Compute incentive utility for cooperative, efficient, trusted behavior."""
    _validate_unit("cooperation_history", cooperation_history)
    _validate_unit("efficiency_score", efficiency_score)
    _validate_unit("effective_trust", effective_trust)
    _validate_sum_one((zeta_coop, zeta_eff, zeta_trust))
    return clip01(
        zeta_coop * cooperation_history
        + zeta_eff * efficiency_score
        + zeta_trust * effective_trust
    )


def total_utility(
    system_u: float,
    social_u: float,
    resource_u: float,
    privacy_u: float,
    incentive_u: float,
    weights: Mapping[str, float],
) -> float:
    """Compute the five-term action utility used for partner selection.

    Fairness is deliberately absent here: candidate-level workload balance is
    logged as a diagnostic, while Jain's fairness index is evaluated over
    realized node contributions at the run level.
    """
    validate_utility_weights(weights)
    values = (system_u, social_u, resource_u, privacy_u, incentive_u)
    for value in values:
        if not math.isfinite(value):
            raise ValueError("utility components must be finite.")
    return (
        weights["system"] * system_u
        + weights["social"] * social_u
        + weights["resource"] * resource_u
        + weights["privacy"] * privacy_u
        + weights["incentive"] * incentive_u
    )


def compute_action_utility(
    score_components: Mapping[str, float | int],
    task: Task,
    requester_resource_score: float,
    target_resource_score: float,
    config: SimulationConfig,
) -> dict[str, float]:
    """Compute all utility components for a candidate action."""
    _validate_unit("requester_resource_score", requester_resource_score)
    _validate_unit("target_resource_score", target_resource_score)
    interaction_probability = float(score_components["interaction_probability"])
    qos = float(score_components["qos"])
    distance_cost = float(score_components["distance_cost"])
    privacy_allowed = int(score_components["privacy_allowed"])
    target_load = clip01(float(score_components.get("target_load", 0.0)))
    expected_success = compute_expected_success(
        interaction_probability,
        qos,
        requester_resource_score,
        target_resource_score,
        config.load_level,
        float(score_components.get("effective_trust", 0.5)),
        task.complexity,
        weights=success_weights_from_config(config),
    )
    energy_cost, bandwidth_cost, compute_cost = compute_resource_costs(
        task.required_bandwidth,
        task.required_compute,
        task.complexity,
        config.load_level,
    )
    delay_ms = config.base_delay_ms * (1.0 + task.complexity + distance_cost)
    delay_norm = clip01(min(delay_ms / 500.0, 1.0))

    system_u = system_utility(
        expected_success,
        delay_norm,
        config.utility_alpha_success,
        config.utility_alpha_delay,
    )
    social_u = social_utility(
        float(score_components["effective_trust"]),
        float(score_components["preference"]),
        float(score_components["reciprocity"]),
        config.utility_beta_trust,
        config.utility_beta_preference,
        config.utility_beta_reciprocity,
    )
    resource_u = resource_utility(
        energy_cost,
        bandwidth_cost,
        compute_cost,
        config.utility_gamma_energy,
        config.utility_gamma_bandwidth,
        config.utility_gamma_compute,
    )
    privacy_u = privacy_utility(
        float(score_components["privacy_risk"]),
        config.utility_delta_privacy,
    )
    fairness_u = fairness_utility(
        target_resource_score,
        target_load,
        config.utility_eta_resource,
        config.utility_eta_load,
    )
    cooperation_history = float(score_components["reciprocity"])
    efficiency_score = clip01((expected_success + resource_u + (1.0 - delay_norm)) / 3.0)
    incentive_u = incentive_utility(
        cooperation_history,
        efficiency_score,
        float(score_components["effective_trust"]),
        config.utility_zeta_coop,
        config.utility_zeta_efficiency,
        config.utility_zeta_trust,
    )
    total_u = total_utility(
        system_u,
        social_u,
        resource_u,
        privacy_u,
        incentive_u,
        _utility_weights_from_config(config),
    )
    return {
        "privacy_allowed": privacy_allowed,
        "interaction_probability": interaction_probability,
        "requester_resource_score": requester_resource_score,
        "target_resource_score": target_resource_score,
        "expected_success": expected_success,
        "delay_ms": delay_ms,
        "delay_norm": delay_norm,
        "energy_cost": energy_cost,
        "bandwidth_cost": bandwidth_cost,
        "compute_cost": compute_cost,
        "system_utility": system_u,
        "social_utility": social_u,
        "resource_utility": resource_u,
        "privacy_utility": privacy_u,
        "fairness_utility": fairness_u,
        "incentive_utility": incentive_u,
        "total_utility": total_u,
    }


def compute_expected_success(
    interaction_probability: float,
    link_quality: float,
    requester_resource_score: float,
    target_resource_score: float,
    load_level: str,
    effective_trust: float = 0.5,
    task_complexity: float = 0.5,
    weights: "SuccessWeights | None" = None,
) -> float:
    """Estimate cooperation success from social, link, resource, and load factors.

    The calibrated model avoids over-penalizing viable interactions by replacing
    the previous multiplicative chain with a weighted additive probability model.
    Load still reduces success through an explicit workload penalty.
    """
    _validate_unit("interaction_probability", interaction_probability)
    _validate_unit("link_quality", link_quality)
    _validate_unit("requester_resource_score", requester_resource_score)
    _validate_unit("target_resource_score", target_resource_score)
    _validate_unit("effective_trust", effective_trust)
    _validate_unit("task_complexity", task_complexity)
    load_factor = {
        "low": 0.00,
        "medium": 0.33,
        "high": 0.66,
        "extreme": 1.00,
    }.get(load_level, 0.05)
    # Intercept of the additive success model: the floor probability before any
    # social, link, resource, load or complexity term applies. Identical for
    # every model -- success depends only on state, never on which policy is
    # being simulated (enforced by tests/test_model_identity_guard.py).
    w = weights or DEFAULT_SUCCESS_WEIGHTS
    return clip01(
        w.base
        + w.probability * interaction_probability
        + w.link_quality * link_quality
        + w.requester_resource * requester_resource_score
        + w.target_resource * target_resource_score
        + w.trust * effective_trust
        - w.load * load_factor
        - w.complexity * task_complexity
    )


def compute_resource_costs(
    required_bandwidth: float,
    required_compute: float,
    complexity: float,
    load_level: str = "medium",
) -> tuple[float, float, float]:
    """Return calibrated per-interaction resource costs.

    Costs remain task-sensitive but are scaled to represent per-step normalized
    service capacity rather than full battery depletion for every interaction.
    """
    _validate_unit("required_bandwidth", required_bandwidth)
    _validate_unit("required_compute", required_compute)
    _validate_unit("complexity", complexity)
    load_multiplier = {
        "low": 0.85,
        "medium": 1.00,
        "high": 1.10,
        "extreme": 1.20,
    }.get(load_level, 1.00)
    scale = 0.70 * load_multiplier
    energy_cost = (0.002 + 0.010 * complexity) * scale
    bandwidth_cost = 0.15 * required_bandwidth * scale
    compute_cost = 0.15 * required_compute * scale
    return clip01(energy_cost), clip01(bandwidth_cost), clip01(compute_cost)


def select_best_utility_partner(
    neighbor_scores: Mapping[int, Mapping[str, float | int]],
) -> int | None:
    """Select allowed neighbor with highest total utility, breaking ties by probability."""
    valid = [
        (
            neighbor_id,
            float(score["total_utility"]),
            float(score["interaction_probability"]),
        )
        for neighbor_id, score in neighbor_scores.items()
        if int(score["privacy_allowed"]) == 1
        and float(score["interaction_probability"]) > 0.0
    ]
    if not valid:
        return None
    return max(valid, key=lambda item: (item[1], item[2]))[0]


def _utility_weights_from_config(config: SimulationConfig) -> dict[str, float]:
    return {
        "system": config.utility_weight_system,
        "social": config.utility_weight_social,
        "resource": config.utility_weight_resource,
        "privacy": config.utility_weight_privacy,
        "incentive": config.utility_weight_incentive,
    }


def _validate_unit(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")


def _validate_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _validate_sum_one(values: tuple[float, ...]) -> None:
    for value in values:
        _validate_nonnegative("component weight", value)
    if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("component weights must sum to approximately 1.0.")

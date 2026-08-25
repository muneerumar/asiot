"""Ablation policies for the proposed Agentic SIoT framework."""

from __future__ import annotations

from typing import Any

from asiot.baselines import (
    _compute_weighted_total,
    _finalize_score,
    PROPOSED_BASELINE_NAME,
    ProposedASIoTFramework,
)
from asiot.datatypes import SimulationConfig, Task
from asiot.node import AgenticNode
from asiot.social_cognition import (
    clip01,
    compute_load_aware_interaction_probability,
)
from asiot.utility import (
    compute_expected_success,
    incentive_utility,
    privacy_utility,
    resource_utility,
    select_best_utility_partner,
    social_utility,
    system_utility,
)

# Stage-0 integrity fix: "without_federated_placeholder" and
# "without_negotiation_placeholder" were removed. Both toggled modules that
# were never implemented, so their rows were byte-identical to full_proposed
# and reporting them as informative ablations was misleading.
ABLATION_VARIANTS = (
    "full_proposed",
    "without_trust",
    "without_preference",
    "without_reciprocity",
    "without_privacy_gate",
    "without_resource_awareness",
    "without_incentive",
    "without_social_graph_adaptation",
)


class AblationPolicy(ProposedASIoTFramework):
    """Proposed-policy variant with one framework component disabled."""

    def __init__(self, variant_name: str) -> None:
        if variant_name not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: {variant_name}")
        self.variant_name = variant_name

    @property
    def name(self) -> str:
        return f"ablation_{self.variant_name}"

    def score_neighbors(
        self,
        node: AgenticNode,
        neighbor_ids: list[int],
        task: Task,
        graph: Any,
        config: SimulationConfig,
        nodes: dict[int, AgenticNode],
    ) -> dict[int, dict[str, float | int | str]]:
        """Score neighbors after applying the selected ablation to proposed scores."""
        proposed_scores = node.compute_neighbor_scores(neighbor_ids, task, graph, config)
        scores = {}
        for neighbor_id, score in proposed_scores.items():
            # Bookkeeping only: ablation runs are logged under the proposed
            # model's name. Nothing on the scoring path reads this field.
            score["baseline_name"] = PROPOSED_BASELINE_NAME
            apply_ablation_to_score(score, self.variant_name, config)
            scores[neighbor_id] = _finalize_score(score)
        return scores

    def select_partner(
        self,
        node: AgenticNode,
        neighbor_scores: dict[int, dict[str, float | int | str]],
        task: Task,
        graph: Any,
        config: SimulationConfig,
    ) -> int | None:
        """Select by utility among allowed candidates."""
        return select_best_utility_partner(neighbor_scores)


def apply_ablation_to_score(
    score: dict[str, float | int | str],
    variant_name: str,
    config: SimulationConfig,
) -> dict[str, float | int | str]:
    """Apply one ablation rule and recompute all dependent utility fields."""
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant_name}")

    if variant_name in {
        "full_proposed",
        "without_social_graph_adaptation",  # behavioral freeze happens in environment.py
    }:
        _recompute_score(score, variant_name, config)
        return score

    if variant_name == "without_trust":
        score["local_trust"] = 0.5
        score["reputation"] = 0.5
        score["effective_trust"] = 0.5
        score["preference"] = clip01(0.5 * float(score["similarity"]) + 0.5 * float(score["qos"]))
    elif variant_name == "without_preference":
        score["preference"] = 0.5
    elif variant_name == "without_reciprocity":
        score["reciprocity"] = 0.5
    elif variant_name == "without_privacy_gate":
        score["privacy_allowed"] = 1
    elif variant_name == "without_resource_awareness":
        # Behavioral ablation: remove resource information from partner
        # selection (probability + expected success), not just the utility
        # bookkeeping. Previously only resource_utility was set to a constant,
        # which never changed rankings because per-candidate resource costs
        # were task-determined and candidate-invariant.
        score["target_resource_score"] = 0.5
        score["requester_resource_score"] = 0.5
        score["resource_utility"] = 0.5
    elif variant_name == "without_incentive":
        score["incentive_utility"] = 0.5

    _recompute_score(score, variant_name, config)
    return score


def compute_ablation_interaction_probability(
    effective_trust: float,
    preference: float,
    reciprocity: float,
    qos: float,
    target_resource_score: float,
    distance_cost: float,
    privacy_allowed: int,
    variant_name: str,
    load_level: str,
    task_complexity: float,
    social_tie: float = 0.5,
) -> float:
    """Compute interaction probability after removing ablated terms."""
    if variant_name == "without_trust":
        effective_trust = 0.5
    if variant_name == "without_preference":
        preference = 0.5
    if variant_name == "without_reciprocity":
        reciprocity = 0.5
    if variant_name == "without_privacy_gate":
        privacy_allowed = 1
    return compute_load_aware_interaction_probability(
        clip01(effective_trust),
        clip01(preference),
        clip01(reciprocity),
        clip01(qos),
        clip01(target_resource_score),
        clip01(distance_cost),
        int(privacy_allowed),
        load_level,
        clip01(task_complexity),
        social_tie=clip01(social_tie),
    )


def _recompute_score(
    score: dict[str, float | int | str],
    variant_name: str,
    config: SimulationConfig,
) -> None:
    score["interaction_probability"] = compute_ablation_interaction_probability(
        float(score["effective_trust"]),
        float(score["preference"]),
        float(score["reciprocity"]),
        float(score["qos"]),
        float(score.get("target_resource_score", 0.5)),
        float(score["distance_cost"]),
        int(score["privacy_allowed"]),
        variant_name,
        config.load_level,
        float(score.get("task_complexity", 0.5)),
        social_tie=float(score.get("social_tie", 0.5)),
    )
    score["expected_success"] = compute_expected_success(
        float(score["interaction_probability"]),
        float(score["qos"]),
        float(score.get("requester_resource_score", 0.5)),
        float(score.get("target_resource_score", 0.5)),
        config.load_level,
        float(score["effective_trust"]),
        float(score.get("task_complexity", 0.5)),
    )
    score["system_utility"] = system_utility(
        float(score["expected_success"]),
        float(score["delay_norm"]),
        config.utility_alpha_success,
        config.utility_alpha_delay,
    )
    if variant_name == "without_trust":
        score["social_utility"] = social_utility(
            0.0,
            float(score["preference"]),
            float(score["reciprocity"]),
        )
    elif variant_name == "without_preference":
        score["social_utility"] = social_utility(
            float(score["effective_trust"]),
            0.0,
            float(score["reciprocity"]),
        )
    elif variant_name == "without_reciprocity":
        score["social_utility"] = social_utility(
            float(score["effective_trust"]),
            float(score["preference"]),
            0.0,
        )
    else:
        score["social_utility"] = social_utility(
            float(score["effective_trust"]),
            float(score["preference"]),
            float(score["reciprocity"]),
            config.utility_beta_trust,
            config.utility_beta_preference,
            config.utility_beta_reciprocity,
        )
    if variant_name != "without_resource_awareness":
        score["resource_utility"] = resource_utility(
            float(score["energy_cost"]),
            float(score["bandwidth_cost"]),
            float(score["compute_cost"]),
            config.utility_gamma_energy,
            config.utility_gamma_bandwidth,
            config.utility_gamma_compute,
        )
    score["privacy_utility"] = privacy_utility(
        float(score["privacy_risk"]),
        config.utility_delta_privacy,
    )
    if variant_name != "without_incentive":
        efficiency_score = clip01(
            (
                float(score["expected_success"])
                + float(score["resource_utility"])
                + (1.0 - float(score["delay_norm"]))
            )
            / 3.0
        )
        score["incentive_utility"] = incentive_utility(
            float(score["reciprocity"]),
            efficiency_score,
            float(score["effective_trust"]),
            config.utility_zeta_coop,
            config.utility_zeta_efficiency,
            config.utility_zeta_trust,
        )
    score["total_utility"] = _compute_weighted_total(score, config)


def is_placeholder_variant(variant_name: str) -> bool:
    """Deprecated: placeholder variants were removed in the Stage-0 integrity fix."""
    return False

"""Metric computation for ASIoT simulation runs."""

from __future__ import annotations

import math
from statistics import mean, pvariance

from asiot.datatypes import InteractionResult, Task
from asiot.node import AgenticNode
from asiot.social_cognition import clip01

EPSILON = 1e-9

STEP_METRIC_COLUMNS = (
    "run_id",
    "seed",
    "load_level",
    "baseline_name",
    "ablation_variant",
    "time_step",
    "generated_tasks",
    "completed_tasks",
    "attempted_interactions",
    "successful_cooperations",
    "failed_interactions",
    "blocked_privacy_interactions",
    "cooperation_rate",
    "task_completion_ratio",
    "reliability_score",
    "resource_efficiency",
    "fairness_index",
    "trust_stability_index",
    "privacy_exposure_risk",
    "candidate_privacy_risk",
    "accepted_privacy_exposure",
    "avg_trust",
    "trust_variance",
    "avg_preference",
    "avg_reciprocity",
    "avg_privacy_risk",
    "avg_total_utility",
    "avg_system_utility",
    "avg_social_utility",
    "avg_resource_utility",
    "avg_privacy_utility",
    "avg_incentive_utility",
    "energy_consumed",
    "bandwidth_consumed",
    "compute_consumed",
    "throughput_mbps",
    "packet_delivery_ratio",
    "e2e_delay_ms",
    "communication_overhead",
    "active_nodes",
)


def cooperation_rate(successful_cooperations: int, attempted_interactions: int) -> float:
    """Return successful cooperations divided by attempted interactions."""
    if attempted_interactions <= 0:
        return 0.0
    return clip01(successful_cooperations / attempted_interactions)


def task_completion_ratio(completed_tasks: int, generated_tasks: int) -> float:
    """Return completed tasks divided by generated tasks."""
    if generated_tasks <= 0:
        return 0.0
    return clip01(completed_tasks / generated_tasks)


def resource_efficiency(
    completed_tasks: int,
    total_energy: float,
    total_bandwidth: float,
    total_compute: float,
    generated_tasks: int | None = None,
) -> float:
    """Return task completion adjusted by normalized resource cost.

    This paper-scale efficiency index rewards completed workload while
    penalizing average resource cost per generated task.
    """
    if generated_tasks is None:
        generated_tasks = max(completed_tasks, 1)
    completion = task_completion_ratio(completed_tasks, generated_tasks)
    weighted_cost = max(
        0.4 * total_energy + 0.3 * total_bandwidth + 0.3 * total_compute,
        0.0,
    )
    normalized_resource_cost = clip01(weighted_cost / max(float(generated_tasks), 1.0))
    return clip01(completion * (1.0 - normalized_resource_cost))


def reliability_score(
    successful_interactions: int,
    failed_interactions: int,
    blocked_interactions: int = 0,
) -> float:
    """Return successful interactions over all resolved interaction outcomes."""
    total = successful_interactions + failed_interactions + blocked_interactions
    if total <= 0:
        return 0.0
    return clip01(successful_interactions / (total + EPSILON))


def fairness_index(values: list[float] | tuple[float, ...]) -> float:
    """Compute Jain's fairness index for nonnegative values."""
    if not values:
        return 0.0
    clipped = [max(0.0, float(value)) for value in values]
    denominator = len(clipped) * sum(value * value for value in clipped)
    if denominator <= 0.0:
        return 0.0
    return clip01((sum(clipped) ** 2) / denominator)


def trust_stability_index(trust_values: list[float] | tuple[float, ...]) -> float:
    """Return one minus normalized trust variance, where higher is more stable."""
    if not trust_values:
        return 0.5
    if len(trust_values) < 2:
        return 1.0
    variance = pvariance([clip01(float(value)) for value in trust_values])
    return clip01(1.0 - min(variance / 0.25, 1.0))


def privacy_exposure_risk(interaction_results: list[InteractionResult]) -> float:
    """Average accepted privacy exposure for attempted allowed interactions."""
    values = [
        result.accepted_privacy_exposure
        if result.accepted_privacy_exposure > 0.0
        else result.privacy_risk
        for result in interaction_results
        if result.attempted and not result.blocked_by_privacy
    ]
    return clip01(mean(values)) if values else 0.0


def candidate_privacy_risk(interaction_results: list[InteractionResult]) -> float:
    """Average privacy risk across all candidate partner sets considered."""
    values = [
        result.candidate_privacy_risk
        for result in interaction_results
        if result.task_id is not None
    ]
    return clip01(mean(values)) if values else 0.0


def throughput_mbps(
    completed_tasks: int,
    time_steps: int,
    task_size_mb: float = 1.0,
) -> float:
    """Return completed task throughput in megabits per simulation step."""
    if time_steps <= 0:
        return 0.0
    return max(0.0, completed_tasks * task_size_mb * 8.0 / time_steps)


def packet_delivery_ratio(successful_messages: int, sent_messages: int) -> float:
    """Return successful messages divided by sent messages."""
    if sent_messages <= 0:
        return 0.0
    return clip01(successful_messages / sent_messages)


def average_end_to_end_delay(interaction_results: list[InteractionResult]) -> float:
    """Average delay in milliseconds for successful attempted interactions."""
    delays = [
        result.delay_ms
        for result in interaction_results
        if result.attempted and result.success
    ]
    return mean(delays) if delays else 0.0


def communication_overhead(control_messages: int, total_messages: int) -> float:
    """Return control-message share when real message counts are supplied.

    The evaluated analytical environment has no packet/message event model and
    therefore records this excluded metric as 0.0 by design.
    """
    if total_messages <= 0:
        return 0.0
    return clip01(control_messages / total_messages)


def summarize_step(
    interaction_results: list[InteractionResult],
    active_nodes: list[AgenticNode],
    generated_tasks: list[Task],
    time_step: int,
    run_id: int,
    seed: int,
    load_level: str,
    baseline_name: str,
    ablation_variant: str = "none",
    participation_counts: dict[int, int] | None = None,
) -> dict[str, float | int | str]:
    """Summarize one environment step with full Phase 6 metrics."""
    completed_tasks = sum(1 for task in generated_tasks if task.completed)
    attempted = [result for result in interaction_results if result.attempted]
    successful = [result for result in interaction_results if result.attempted and result.success]
    failed = [
        result
        for result in interaction_results
        if result.attempted and not result.success and not result.blocked_by_privacy
    ]
    blocked = [result for result in interaction_results if result.blocked_by_privacy]
    privacy_values = [result.privacy_risk for result in interaction_results]
    total_utilities = [result.total_utility for result in attempted]
    system_utilities = [result.system_utility for result in attempted]
    social_utilities = [result.social_utility for result in attempted]
    resource_utilities = [result.resource_utility for result in attempted]
    privacy_utilities = [result.privacy_utility for result in attempted]
    incentive_utilities = [result.incentive_utility for result in attempted]
    energy = sum(result.energy_cost for result in interaction_results)
    bandwidth = sum(result.bandwidth_cost for result in interaction_results)
    compute = sum(result.compute_cost for result in interaction_results)
    if participation_counts is None:
        participation_values = [0.0 for _ in active_nodes]
    else:
        participation_values = [
            float(participation_counts.get(node.state.node_id, 0))
            for node in active_nodes
        ]
    trust_values = _collect_social_values(active_nodes, "trust")

    return {
        "run_id": run_id,
        "seed": seed,
        "load_level": load_level,
        "baseline_name": baseline_name,
        "ablation_variant": ablation_variant,
        "time_step": time_step,
        "generated_tasks": len(generated_tasks),
        "completed_tasks": completed_tasks,
        "attempted_interactions": len(attempted),
        "successful_cooperations": len(successful),
        "failed_interactions": len(failed),
        "blocked_privacy_interactions": len(blocked),
        "cooperation_rate": cooperation_rate(len(successful), len(attempted)),
        "task_completion_ratio": task_completion_ratio(completed_tasks, len(generated_tasks)),
        "reliability_score": reliability_score(len(successful), len(failed), len(blocked)),
        "resource_efficiency": resource_efficiency(
            completed_tasks,
            energy,
            bandwidth,
            compute,
            generated_tasks=len(generated_tasks),
        ),
        "fairness_index": fairness_index(participation_values),
        "trust_stability_index": trust_stability_index(trust_values),
        "privacy_exposure_risk": privacy_exposure_risk(interaction_results),
        "candidate_privacy_risk": candidate_privacy_risk(interaction_results),
        "accepted_privacy_exposure": privacy_exposure_risk(interaction_results),
        "avg_trust": compute_average_social_value(active_nodes, "trust"),
        "trust_variance": compute_trust_variance(active_nodes),
        "avg_preference": compute_average_social_value(active_nodes, "preference"),
        "avg_reciprocity": compute_average_social_value(active_nodes, "reciprocity"),
        "avg_privacy_risk": mean(privacy_values) if privacy_values else 0.0,
        "avg_total_utility": mean(total_utilities) if total_utilities else 0.0,
        "avg_system_utility": mean(system_utilities) if system_utilities else 0.0,
        "avg_social_utility": mean(social_utilities) if social_utilities else 0.0,
        "avg_resource_utility": mean(resource_utilities) if resource_utilities else 0.0,
        "avg_privacy_utility": mean(privacy_utilities) if privacy_utilities else 0.0,
        "avg_incentive_utility": mean(incentive_utilities) if incentive_utilities else 0.0,
        "energy_consumed": energy,
        "bandwidth_consumed": bandwidth,
        "compute_consumed": compute,
        "throughput_mbps": throughput_mbps(completed_tasks, 1),
        "packet_delivery_ratio": packet_delivery_ratio(len(successful), len(attempted)),
        "e2e_delay_ms": average_end_to_end_delay(interaction_results),
        "communication_overhead": 0.0,
        "active_nodes": len(active_nodes),
    }


def compute_trust_variance(nodes: list[AgenticNode]) -> float:
    """Compute population variance over all local trust values."""
    values = _collect_social_values(nodes, "trust")
    if len(values) < 2:
        return 0.0
    return pvariance(values)


def compute_average_social_value(nodes: list[AgenticNode], field_name: str) -> float:
    """Compute mean over trust, preference, or reciprocity social-state entries."""
    if field_name not in {"trust", "preference", "reciprocity"}:
        raise ValueError("field_name must be trust, preference, or reciprocity.")
    values = _collect_social_values(nodes, field_name)
    return clip01(mean(values)) if values else 0.5


def _collect_social_values(nodes: list[AgenticNode], field_name: str) -> list[float]:
    values = []
    for node in nodes:
        field = getattr(node.state.social, field_name)
        values.extend(clip01(float(value)) for value in field.values())
    return values


def is_finite_number(value: object) -> bool:
    """Return whether value can be interpreted as a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

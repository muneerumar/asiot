"""Baseline policies for comparing against the proposed A-SIoT framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from asiot.datatypes import SimulationConfig, Task
from asiot.node import AgenticNode
from asiot.social_cognition import (
    clip01,
    compute_effective_trust,
    compute_interaction_probability,
    compute_load_aware_interaction_probability,
    compute_preference,
    compute_privacy_risk,
    privacy_gate,
    sigmoid,
)
from asiot.nitti_trust import (
    WINDOW_LONG,
    WINDOW_RECENT,
    FeedbackLedger,
    capability as nitti_capability,
    centrality as nitti_centrality,
    credibility as nitti_credibility,
    direct_opinion as nitti_direct_opinion,
    indirect_opinion as nitti_indirect_opinion,
    relationship_factor as nitti_relationship_factor,
    subjective_trustworthiness as nitti_subjective_trustworthiness,
)
from asiot.utility import (
    compute_action_utility,
    select_best_utility_partner,
    social_utility,
    total_utility,
)

REQUIRED_SCORE_KEYS = (
    "local_trust",
    "reputation",
    "effective_trust",
    "similarity",
    "qos",
    "preference",
    "reciprocity",
    "privacy_risk",
    "privacy_allowed",
    "distance_cost",
    "interaction_probability",
    "expected_success",
    "delay_ms",
    "delay_norm",
    "energy_cost",
    "bandwidth_cost",
    "compute_cost",
    "system_utility",
    "social_utility",
    "resource_utility",
    "privacy_utility",
    "fairness_utility",
    "incentive_utility",
    "total_utility",
    "baseline_name",
)


def ensure_required_score_keys(score: dict[str, float | int | str]) -> None:
    """Raise ValueError when a baseline score misses required common fields."""
    missing = set(REQUIRED_SCORE_KEYS).difference(score)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"baseline score is missing required keys: {missing_names}")


def _finalize_score(score: dict[str, float | int | str]) -> dict[str, float | int | str]:
    """Validate and return only the common baseline score fields."""
    ensure_required_score_keys(score)
    return {key: score[key] for key in REQUIRED_SCORE_KEYS}


def _compute_weighted_total(
    score: dict[str, float | int | str],
    config: SimulationConfig,
) -> float:
    """Compute the canonical five-term action utility."""
    return total_utility(
        float(score["system_utility"]),
        float(score["social_utility"]),
        float(score["resource_utility"]),
        float(score["privacy_utility"]),
        float(score["incentive_utility"]),
        {
            "system": config.utility_weight_system,
            "social": config.utility_weight_social,
            "resource": config.utility_weight_resource,
            "privacy": config.utility_weight_privacy,
            "incentive": config.utility_weight_incentive,
        },
    )


class BaselinePolicy(ABC):
    """Base interface for baseline decision policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable baseline name used in scripts, CSVs, and docs."""

    @abstractmethod
    def score_neighbors(
        self,
        node: AgenticNode,
        neighbor_ids: list[int],
        task: Task,
        graph: Any,
        config: SimulationConfig,
        nodes: dict[int, AgenticNode],
    ) -> dict[int, dict[str, float | int | str]]:
        """Score candidate neighbors for one task."""

    def select_partner(
        self,
        node: AgenticNode,
        neighbor_scores: dict[int, dict[str, float | int | str]],
        task: Task,
        graph: Any,
        config: SimulationConfig,
    ) -> int | None:
        """Select a partner from scored neighbors."""
        return select_best_utility_partner(neighbor_scores)

    def observe_outcome(
        self,
        requester_id: int,
        partner_id: int,
        success: bool,
        time_step: int,
    ) -> None:
        """Receive the outcome of a completed interaction.

        A no-op for every policy that derives its trust from the shared node
        state. Feedback-history models -- Nitti et al. (2014) keeps windowed
        per-transaction feedback -- override this to record the transaction.
        The hook carries no randomness and returns nothing, so a policy that
        ignores it behaves exactly as before.
        """

    def reset(self) -> None:
        """Clear policy-owned state before a new environment episode."""

    def on_identity_reset(self, node_id: int) -> None:
        """Clear policy-owned history involving a whitewashing identity."""

    def _base_components(
        self,
        node: AgenticNode,
        neighbor_id: int,
        task: Task,
        graph: Any,
        config: SimulationConfig,
        nodes: dict[int, AgenticNode],
    ) -> dict[str, float | int | str]:
        local_trust = node.state.social.trust.get(neighbor_id, config.initial_trust)
        reputation = graph.get_node_reputation(neighbor_id)
        effective_trust = compute_effective_trust(local_trust, reputation, mu=0.65)
        similarity = graph.get_domain_similarity(node.state.node_id, neighbor_id)
        qos = graph.get_link_quality(node.state.node_id, neighbor_id)
        reciprocity = node.state.social.reciprocity.get(neighbor_id, 0.5)
        target_exposure = graph.get_node_exposure_history(neighbor_id)
        privacy_risk = compute_privacy_risk(
            task.data_sensitivity,
            target_exposure,
            reputation,
            weights=(0.45, 0.25, 0.30),
        )
        preference = compute_preference(
            effective_trust,
            similarity,
            qos,
            weights=(0.5, 0.25, 0.25),
        )
        privacy_allowed = privacy_gate(privacy_risk, node.state.social.privacy_threshold)
        distance_cost = graph.get_distance_cost(node.state.node_id, neighbor_id)
        target_resource_score = nodes[neighbor_id].get_resource_score()
        probability = compute_load_aware_interaction_probability(
            effective_trust,
            preference,
            reciprocity,
            qos,
            target_resource_score,
            distance_cost,
            privacy_allowed,
            config.load_level,
            task.complexity,
        )
        components: dict[str, float | int | str] = {
            "baseline_name": self.name,
            "local_trust": local_trust,
            "reputation": reputation,
            "effective_trust": effective_trust,
            "similarity": similarity,
            "qos": qos,
            "preference": preference,
            "reciprocity": reciprocity,
            "privacy_risk": privacy_risk,
            "privacy_allowed": privacy_allowed,
            "distance_cost": distance_cost,
            "interaction_probability": probability,
            "requester_resource_score": node.get_resource_score(),
            "target_resource_score": target_resource_score,
            "task_complexity": task.complexity,
            "target_load": clip01(1.0 - target_resource_score),
        }
        components.update(
            compute_action_utility(
                components,
                task,
                node.get_resource_score(),
                target_resource_score,
                config,
            )
        )
        components["total_utility"] = _compute_weighted_total(components, config)
        return components

    def _score_all(
        self,
        node: AgenticNode,
        neighbor_ids: list[int],
        task: Task,
        graph: Any,
        config: SimulationConfig,
        nodes: dict[int, AgenticNode],
    ) -> dict[int, dict[str, float | int | str]]:
        return {
            neighbor_id: self._base_components(node, neighbor_id, task, graph, config, nodes)
            for neighbor_id in neighbor_ids
        }


#: Registry name of the full proposed framework. Model identity belongs to the
#: policy layer, which is why it is defined here rather than in a module that
#: computes scores -- see tests/test_model_identity_guard.py.
PROPOSED_BASELINE_NAME = "proposed"


class ProposedASIoTFramework(BaselinePolicy):
    """Full proposed model using social cognition, privacy, and total utility."""

    @property
    def name(self) -> str:
        return PROPOSED_BASELINE_NAME

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = node.compute_neighbor_scores(neighbor_ids, task, graph, config)
        for score in scores.values():
            score["baseline_name"] = self.name
            score["total_utility"] = _compute_weighted_total(score, config)
        return {neighbor_id: _finalize_score(score) for neighbor_id, score in scores.items()}


class NonAgenticStaticBaseline(BaselinePolicy):
    """Static non-agentic IoT/SIoT baseline without social intelligence."""

    @property
    def name(self) -> str:
        return "non_agentic_static"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            qos = float(score["qos"])
            score.update(
                {
                    "local_trust": 0.5,
                    "reputation": 0.5,
                    "effective_trust": 0.5,
                    "preference": 0.5,
                    "reciprocity": 0.5,
                    "privacy_allowed": 1,
                    "privacy_risk": 0.5,
                    "interaction_probability": clip01(qos * 0.5),
                }
            )
            score.update(
                compute_action_utility(
                    score,
                    task,
                    node.get_resource_score(),
                    nodes[neighbor_id].get_resource_score(),
                    config,
                )
            )
            score["total_utility"] = score["expected_success"]
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores

    def select_partner(self, node, neighbor_scores, task, graph, config):
        for neighbor_id, score in neighbor_scores.items():
            if int(score["privacy_allowed"]) == 1 and float(score["interaction_probability"]) > 0.0:
                return neighbor_id
        return None


class HonestyBasedSocialModel(BaselinePolicy):
    """Simple honesty/trust-like social baseline."""

    @property
    def name(self) -> str:
        return "honesty_based_social"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            effective = compute_effective_trust(
                float(score["local_trust"]),
                float(score["reputation"]),
                mu=0.65,
            )
            probability = sigmoid(
                1.2 * effective
                + 0.5 * float(score["qos"])
                - 0.3 * float(score["distance_cost"])
            )
            score.update(
                {
                    "effective_trust": effective,
                    "preference": effective,
                    "reciprocity": 0.5,
                    "privacy_allowed": 1,
                    "privacy_risk": 0.5,
                    "interaction_probability": probability,
                }
            )
            score.update(
                compute_action_utility(
                    score,
                    task,
                    node.get_resource_score(),
                    nodes[neighbor_id].get_resource_score(),
                    config,
                )
            )
            score["total_utility"] = clip01(0.7 * effective + 0.3 * float(score["expected_success"]))
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores


class GreedyUtilityBasedModel(BaselinePolicy):
    """System-performance-only greedy utility baseline."""

    @property
    def name(self) -> str:
        return "greedy_utility"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            target_resource = nodes[neighbor_id].get_resource_score()
            probability = clip01(float(score["qos"]) * node.get_resource_score() * target_resource)
            score.update(
                {
                    "local_trust": 0.5,
                    "reputation": 0.5,
                    "effective_trust": 0.5,
                    "preference": float(score["qos"]),
                    "reciprocity": 0.5,
                    "privacy_allowed": 1,
                    "privacy_risk": 0.5,
                    "interaction_probability": probability,
                }
            )
            score.update(
                compute_action_utility(score, task, node.get_resource_score(), target_resource, config)
            )
            score["total_utility"] = clip01(
                0.6 * float(score["expected_success"])
                + 0.3 * float(score["resource_utility"])
                + 0.1 * float(score["system_utility"])
            )
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores


class StandardMARLNoSocialModel(BaselinePolicy):
    """Deterministic non-social heuristic retained under its legacy API name."""

    @property
    def name(self) -> str:
        return "standard_marl_no_social"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            target_resource = nodes[neighbor_id].get_resource_score()
            probability = sigmoid(
                float(score["qos"])
                + 0.8 * target_resource
                - 0.6 * float(score["distance_cost"])
                - 0.4 * task.complexity
            )
            score.update(
                {
                    "local_trust": 0.5,
                    "reputation": 0.5,
                    "effective_trust": 0.5,
                    "preference": 0.5,
                    "reciprocity": 0.5,
                    "privacy_allowed": 1,
                    "privacy_risk": 0.5,
                    "interaction_probability": probability,
                }
            )
            score.update(
                compute_action_utility(score, task, node.get_resource_score(), target_resource, config)
            )
            score["total_utility"] = clip01(
                float(score["expected_success"])
                - 0.2 * float(score["delay_norm"])
                + 0.2 * float(score["resource_utility"])
            )
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores


class NeuralMARLNoSocialModel(StandardMARLNoSocialModel):
    """Trained neural MARL baseline without social features.

    Evaluation with a checkpoint is executed through
    ``ASIoTEnvironment.step_with_neural_policy``. The score function remains
    available so the environment can build normalized candidate observations.
    """

    @property
    def name(self) -> str:
        return "neural_marl_no_social"


class NeuralMARLSocialModel(ProposedASIoTFramework):
    """Trained neural MARL policy using social candidate features.

    Evaluation with a checkpoint is executed through
    ``ASIoTEnvironment.step_with_neural_policy``. The score function reuses the
    proposed social-cognitive candidate model as neural input features.
    """

    @property
    def name(self) -> str:
        return "neural_marl_social"


class TrustUnawareModel(BaselinePolicy):
    """Baseline that disables trust and reputation while keeping other mechanisms."""

    @property
    def name(self) -> str:
        return "trust_unaware"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            preference = clip01(0.5 * float(score["similarity"]) + 0.5 * float(score["qos"]))
            score.update(
                {
                    "local_trust": 0.5,
                    "reputation": 0.5,
                    "effective_trust": 0.5,
                    "preference": preference,
                }
            )
            probability = compute_interaction_probability(
                preference,
                float(score["reciprocity"]),
                float(score["distance_cost"]),
                int(score["privacy_allowed"]),
                alpha=1.0,
                beta=0.8,
                gamma=0.8,
            )
            score["interaction_probability"] = probability
            score.update(
                compute_action_utility(
                    score,
                    task,
                    node.get_resource_score(),
                    nodes[neighbor_id].get_resource_score(),
                    config,
                )
            )
            score["social_utility"] = social_utility(0.0, preference, float(score["reciprocity"]))
            score["total_utility"] = clip01(
                0.30 * float(score["system_utility"])
                + 0.25 * float(score["resource_utility"])
                + 0.25 * float(score["privacy_utility"])
                + 0.20 * float(score["incentive_utility"])
            )
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores


class GameTheoreticSocialModel(BaselinePolicy):
    """Strategic social interaction baseline with payoff-like scoring."""

    @property
    def name(self) -> str:
        return "game_theoretic_social"

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        for neighbor_id, score in scores.items():
            effective = compute_effective_trust(
                float(score["local_trust"]),
                float(score["reputation"]),
                mu=0.65,
            )
            reciprocity = float(score["reciprocity"])
            preference = clip01(0.5 * effective + 0.3 * reciprocity + 0.2 * float(score["qos"]))
            score.update(
                {
                    "effective_trust": effective,
                    "preference": preference,
                    "privacy_allowed": 1,
                }
            )
            score["interaction_probability"] = sigmoid(
                preference + reciprocity - 0.5 * float(score["distance_cost"])
            )
            score.update(
                compute_action_utility(
                    score,
                    task,
                    node.get_resource_score(),
                    nodes[neighbor_id].get_resource_score(),
                    config,
                )
            )
            payoff = (
                0.4 * effective
                + 0.3 * reciprocity
                + 0.2 * float(score["expected_success"])
                - 0.1 * float(score["distance_cost"])
            )
            score["total_utility"] = clip01(payoff)
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores



class NittiSubjectiveTrustModel(BaselinePolicy):
    """Nitti, Girau & Atzori (2014) subjective trustworthiness, IEEE TKDE 26(5).

    External state-of-the-art baseline. Partner ranking is driven by the
    paper's T_ij (Eq. 1), computed from centrality (Eq. 2), the node's own
    windowed feedback history (Eqs. 3-5) and the credibility-weighted opinion
    of common friends (Eqs. 6-7). See src/asiot/nitti_trust.py for the
    equations and docs/nitti_assumptions.md for every choice the paper leaves
    open.

    The surrounding scoring skeleton (QoS, distance, resource terms, utility
    aggregation) is identical to the other social baselines, so the comparison
    isolates the trust model rather than the harness around it.
    """

    def __init__(self) -> None:
        # One ledger per policy instance, and one policy instance per
        # environment, so feedback never leaks across runs or seeds.
        self.ledger = FeedbackLedger()

    @property
    def name(self) -> str:
        return "nitti_subjective_trust"

    def observe_outcome(self, requester_id, partner_id, success, time_step) -> None:
        """Record binary feedback f_l for the completed transaction."""
        self.ledger.record(requester_id, partner_id, success)

    def reset(self) -> None:
        """Clear all transaction feedback before a new episode."""
        self.ledger.clear()

    def on_identity_reset(self, node_id: int) -> None:
        """Forget transactions to or from a whitewashing identity."""
        self.ledger.reset_node(node_id)

    def _direct_opinion(self, source_id, target_id, graph, nodes) -> float:
        """Eq. (3) for an arbitrary ordered pair."""
        transactions = self.ledger.transactions(source_id, target_id)
        return nitti_direct_opinion(
            transactions,
            self.ledger.opinion(source_id, target_id, WINDOW_LONG),
            self.ledger.opinion(source_id, target_id, WINDOW_RECENT),
            nitti_relationship_factor(
                nodes[source_id].state.domain, nodes[target_id].state.domain
            ),
            nitti_capability(nodes[target_id].state.role),
        )

    def _centrality(self, source_id, target_id, graph) -> float:
        """Eq. (2) from the live social graph."""
        source_neighbours = set(graph.get_neighbors(source_id))
        target_neighbours = set(graph.get_neighbors(target_id))
        common = source_neighbours & target_neighbours
        return nitti_centrality(len(common), len(source_neighbours))

    def score_neighbors(self, node, neighbor_ids, task, graph, config, nodes):
        scores = self._score_all(node, neighbor_ids, task, graph, config, nodes)
        source_id = node.state.node_id
        source_neighbours = set(graph.get_neighbors(source_id))
        for neighbor_id, score in scores.items():
            centrality_ij = self._centrality(source_id, neighbor_id, graph)
            direct_ij = self._direct_opinion(source_id, neighbor_id, graph, nodes)

            # Eq. (6): common friends K_ij weigh in, each weighted by Eq. (7).
            common_friends = source_neighbours & set(graph.get_neighbors(neighbor_id))
            contributions = []
            for friend_id in sorted(common_friends):
                if friend_id in (source_id, neighbor_id) or friend_id not in nodes:
                    continue
                credibility_ik = nitti_credibility(
                    self._direct_opinion(source_id, friend_id, graph, nodes),
                    self._centrality(source_id, friend_id, graph),
                )
                contributions.append(
                    (credibility_ik,
                     self._direct_opinion(friend_id, neighbor_id, graph, nodes))
                )
            indirect_ij = nitti_indirect_opinion(
                contributions, fallback=config.nitti_empty_kij_fallback
            )

            trustworthiness = clip01(
                nitti_subjective_trustworthiness(centrality_ij, direct_ij, indirect_ij)
            )

            probability = sigmoid(
                1.2 * trustworthiness
                + 0.5 * float(score["qos"])
                - 0.3 * float(score["distance_cost"])
            )
            score.update(
                {
                    "effective_trust": trustworthiness,
                    "preference": trustworthiness,
                    "reciprocity": 0.5,
                    "privacy_allowed": 1,
                    "privacy_risk": 0.5,
                    "interaction_probability": probability,
                    "nitti_centrality": centrality_ij,
                    "nitti_direct_opinion": direct_ij,
                    "nitti_indirect_opinion": indirect_ij,
                }
            )
            score.update(
                compute_action_utility(
                    score,
                    task,
                    node.get_resource_score(),
                    nodes[neighbor_id].get_resource_score(),
                    config,
                )
            )
            # Ranking is by T_ij, which is what the paper's node selects on:
            # "a node chooses the provider of the service on the basis of the
            # highest computed trustworthiness level".
            score["total_utility"] = clip01(
                0.7 * trustworthiness + 0.3 * float(score["expected_success"])
            )
            score["baseline_name"] = self.name
            scores[neighbor_id] = _finalize_score(score)
        return scores


BASELINE_REGISTRY = {
    "non_agentic_static": NonAgenticStaticBaseline,
    "honesty_based_social": HonestyBasedSocialModel,
    "greedy_utility": GreedyUtilityBasedModel,
    "standard_marl_no_social": StandardMARLNoSocialModel,
    "neural_marl_no_social": NeuralMARLNoSocialModel,
    "neural_marl_social": NeuralMARLSocialModel,
    "trust_unaware": TrustUnawareModel,
    "game_theoretic_social": GameTheoreticSocialModel,
    "nitti_subjective_trust": NittiSubjectiveTrustModel,
    "proposed": ProposedASIoTFramework,
}


def get_baseline_policy(name: str) -> BaselinePolicy:
    """Return a baseline policy by stable registry name."""
    try:
        return BASELINE_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown baseline policy: {name}") from exc

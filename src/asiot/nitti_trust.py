"""Subjective trustworthiness model of Nitti, Girau & Atzori (2014).

Faithful implementation of the SUBJECTIVE model from:

    M. Nitti, R. Girau, L. Atzori, "Trustworthiness Management in the Social
    Internet of Things," IEEE Transactions on Knowledge and Data Engineering,
    26(5):1253-1266, 2014.

Equation numbers below are the paper's own. The subjective model is the one
implemented because it is the socially-grounded variant (each node keeps its
own feedback and combines it with common-friend opinion); the objective model
(Eqs. 10-12) presumes a DHT of pre-trusted objects and a network-wide view,
which this simulator does not provide.

    (1)  T_ij      = (1 - a - b) R_ij + a O_dir_ij + b O_ind_ij
    (2)  R_ij      = |K_ij| / (|N_i| - 1)
    (3)  O_dir_ij  = [ log(N_ij+1) / (1 + log(N_ij+1)) ] (g O_lon_ij + (1-g) O_rec_ij)
                   + [ 1 / (1 + log(N_ij+1)) ] (d F_ij + (1-d)(1 - I_j))
    (4)  O_lon_ij  = sum_{l=1..Llon} w_l f_l / sum_{l=1..Llon} w_l
    (5)  O_rec_ij  = sum_{l=1..Lrec} w_l f_l / sum_{l=1..Lrec} w_l
    (6)  O_ind_ij  = sum_{k in K_ij} C_ik O_dir_kj / sum_{k in K_ij} C_ik
    (7)  C_ik      = e O_dir_ik + (1 - e) R_ik

Paper defaults, Table V (subjective model): a=0.4, b=0.3, g=0.5, d=0.5,
e=0.7, Llon=50, Lrec=5. Table IV supplies the relationship factor F and the
computation capability I. The authors set the transaction factor w_l = 1 for
all transactions and use binary feedback; both choices are theirs, not ours.

Every place where the paper leaves something unspecified for a simulator like
ours is marked ASSUMPTION and listed in docs/nitti_assumptions.md.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# --- Table V: subjective model parameters (paper's optimal configuration) ---
ALPHA_DIRECT = 0.4      # a: weight of the direct opinion
BETA_INDIRECT = 0.3     # b: weight of the indirect opinion
GAMMA_LONG = 0.5        # g: weight of the long-term opinion
DELTA_RELATIONSHIP = 0.5  # d: weight of the relationship factor
ETA_CREDIBILITY = 0.7   # e: weight of the direct opinion in the credibility
WINDOW_LONG = 50        # Llon
WINDOW_RECENT = 5       # Lrec
TRANSACTION_FACTOR = 1.0  # w_l; the paper sets all transactions equally relevant

# --- Table IV: relationship factor F_ij ---
RELATIONSHIP_FACTOR = {
    "OOR": 1.0,    # Ownership Object Relationship
    "CLOR": 0.8,   # Co-Location Object Relationship
    "CWOR": 0.8,   # Co-Work Object Relationship
    "SOR": 0.6,    # Social Object Relationship
    "POR": 0.5,    # Parental Object Relationship
}

# --- Table IV: computation capability I_j ---
CAPABILITY_CLASS_1 = 0.8  # smartphone, tablet, set-top box
CAPABILITY_CLASS_2 = 0.2  # sensor, RFID

# ASSUMPTION (A1). This simulator has no SIoT relationship taxonomy
# (OOR/CLOR/CWOR/SOR/POR). Nodes carry a `domain` instead. We map same-domain
# pairs to CWOR (0.8) -- objects working in the same domain are co-workers --
# and cross-domain pairs to SOR (0.6), the generic social relation. OOR is
# unreachable because no node owns another here, and POR is unreachable
# because there is no manufacturer/batch notion. The chosen pair spans the
# middle of the paper's range rather than its extremes.
RELATIONSHIP_SAME_DOMAIN = RELATIONSHIP_FACTOR["CWOR"]
RELATIONSHIP_CROSS_DOMAIN = RELATIONSHIP_FACTOR["SOR"]

# ASSUMPTION (A2). Computation capability maps onto this simulator's node
# `role`, which is otherwise unused. Nitti's Class 2 is "any object just
# capable of providing a measure of the environment status"; `sensor` is
# exactly that. The remaining roles (relay, coordinator, actuator) are
# programmable devices and take Class 1. Note the paper's own convention that
# HIGHER capability LOWERS trust -- Eq. (3) uses (1 - I_j) -- because a smarter
# object is better able to cheat.
CLASS_2_ROLES = frozenset({"sensor"})


def capability(role: str) -> float:
    """Return I_j from Table IV for a node role (see ASSUMPTION A2)."""
    return CAPABILITY_CLASS_2 if role in CLASS_2_ROLES else CAPABILITY_CLASS_1


def relationship_factor(domain_i: str, domain_j: str) -> float:
    """Return F_ij from Table IV (see ASSUMPTION A1)."""
    return RELATIONSHIP_SAME_DOMAIN if domain_i == domain_j else RELATIONSHIP_CROSS_DOMAIN


@dataclass
class FeedbackLedger:
    """Per-pair windowed feedback history, as Eqs. (4)-(5) require.

    The paper's model is defined over the last Llon transactions between a
    specific ordered pair, which this simulator's aggregate interaction counts
    cannot supply. The ledger is owned by the policy instance (one per
    environment), so it neither leaks between runs nor perturbs shared state.
    """

    history: dict[tuple[int, int], deque[float]] = field(default_factory=dict)

    def record(self, requester_id: int, partner_id: int, success: bool) -> None:
        """Append binary feedback f_l for one transaction (paper: f in {0,1})."""
        key = (requester_id, partner_id)
        window = self.history.setdefault(key, deque(maxlen=WINDOW_LONG))
        window.appendleft(1.0 if success else 0.0)

    def clear(self) -> None:
        """Remove all retained feedback."""
        self.history.clear()

    def reset_node(self, node_id: int) -> None:
        """Remove every ordered-pair history involving ``node_id``."""
        for pair in [pair for pair in self.history if node_id in pair]:
            del self.history[pair]

    def transactions(self, requester_id: int, partner_id: int) -> int:
        """N_ij: total transactions retained between the pair."""
        return len(self.history.get((requester_id, partner_id), ()))

    def opinion(self, requester_id: int, partner_id: int, window: int) -> float:
        """Eqs. (4)/(5): w-weighted mean feedback over the newest `window` items.

        With w_l = 1 for all l (the paper's setting) this is the plain mean of
        the retained feedback. Returns 0.0 for an empty history; Eq. (3) gates
        that case to zero weight through the log(N_ij+1) term, so the value is
        never actually used.
        """
        entries = self.history.get((requester_id, partner_id))
        if not entries:
            return 0.0
        recent = list(entries)[:window]
        weight_total = TRANSACTION_FACTOR * len(recent)
        if weight_total <= 0.0:
            return 0.0
        return sum(TRANSACTION_FACTOR * f for f in recent) / weight_total


def centrality(common_friends: int, neighbour_count: int) -> float:
    """Eq. (2): R_ij = |K_ij| / (|N_i| - 1).

    ASSUMPTION (A3): the paper does not define R_ij when |N_i| = 1, where the
    denominator vanishes. We return 0.0 -- a node with a single friend has no
    basis for judging shared social position. Our graph enforces
    min_neighbors >= 3, so this branch is defensive rather than routine.
    """
    denominator = neighbour_count - 1
    if denominator <= 0:
        return 0.0
    return min(1.0, common_friends / denominator)


def direct_opinion(
    transactions: int,
    long_term: float,
    recent: float,
    relationship: float,
    capability_value: float,
) -> float:
    """Eq. (3): experience-weighted opinion, falling back to F and I."""
    log_term = math.log(transactions + 1)
    experience_weight = log_term / (1.0 + log_term)
    prior_weight = 1.0 / (1.0 + log_term)
    experience = GAMMA_LONG * long_term + (1.0 - GAMMA_LONG) * recent
    prior = (DELTA_RELATIONSHIP * relationship
             + (1.0 - DELTA_RELATIONSHIP) * (1.0 - capability_value))
    return experience_weight * experience + prior_weight * prior


def credibility(direct_opinion_ik: float, centrality_ik: float) -> float:
    """Eq. (7): C_ik = e O_dir_ik + (1 - e) R_ik."""
    return ETA_CREDIBILITY * direct_opinion_ik + (1.0 - ETA_CREDIBILITY) * centrality_ik


EMPTY_KIJ_FALLBACK = 0.5


def indirect_opinion(
    contributions: list[tuple[float, float]],
    fallback: float = EMPTY_KIJ_FALLBACK,
) -> float:
    """Eq. (6): credibility-weighted mean of common friends' direct opinions.

    ``contributions`` is a list of (C_ik, O_dir_kj) pairs over k in K_ij.

    ASSUMPTION (A4): the paper does not state the value of O_ind_ij when
    K_ij is empty or all credibilities are zero. We return 0.5 -- maximal
    ignorance on the model's [0, 1] scale -- rather than 0.0, which would be
    indistinguishable from unanimous condemnation by common friends.
    """
    if not contributions:
        return fallback
    weight_total = sum(weight for weight, _ in contributions)
    if weight_total <= 0.0:
        return fallback
    return sum(weight * value for weight, value in contributions) / weight_total


def subjective_trustworthiness(
    centrality_ij: float,
    direct_ij: float,
    indirect_ij: float,
) -> float:
    """Eq. (1): T_ij = (1 - a - b) R_ij + a O_dir_ij + b O_ind_ij."""
    return ((1.0 - ALPHA_DIRECT - BETA_INDIRECT) * centrality_ij
            + ALPHA_DIRECT * direct_ij
            + BETA_INDIRECT * indirect_ij)

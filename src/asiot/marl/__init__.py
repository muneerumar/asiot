"""Neural MARL/DRL extension namespace for Agentic A-SIoT."""

from asiot.marl.dqn_agent import DQNAgent
from asiot.marl.observations import OBSERVATION_DIM, build_candidate_observation
from asiot.marl.policy_network import DQNPolicyNetwork
from asiot.marl.replay_buffer import ReplayBuffer, Transition

__all__ = [
    "DQNAgent",
    "DQNPolicyNetwork",
    "OBSERVATION_DIM",
    "ReplayBuffer",
    "Transition",
    "build_candidate_observation",
]

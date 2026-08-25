"""Configuration loading for the Agentic SIoT simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from asiot.datatypes import SimulationConfig


def load_config(path: Path | str) -> SimulationConfig:
    """Load a simulation configuration from a YAML file.

    Args:
        path: Path to a configuration file.

    Returns:
        A parsed simulation configuration.
    """
    config_path = Path(path)
    data = _read_yaml(config_path)

    simulation = _section(data, "simulation")
    network = _section(data, "network")
    social = _section(data, "social")
    privacy = _section(data, "privacy")
    resources = _section(data, "resources")
    communication = _section(data, "communication")
    deployment = _section(data, "deployment")
    outputs = _section(data, "outputs")
    tasks = _section(data, "tasks")
    utility = _section(data, "utility")
    utility_weights = _section(utility, "weights")
    utility_system = _section(utility, "system")
    utility_social = _section(utility, "social")
    utility_resource = _section(utility, "resource")
    utility_privacy = _section(utility, "privacy")
    utility_fairness = _section(utility, "fairness")
    utility_incentive = _section(utility, "incentive")

    return SimulationConfig(
        name=str(simulation.get("name", "default")),
        random_seed=int(simulation.get("random_seed", 42)),
        steps=int(simulation.get("steps", 100)),
        node_count=int(network.get("node_count", 50)),
        domains=tuple(
            deployment.get(
                "domains",
                ["smart_home", "healthcare", "transportation", "utilities"],
            )
        ),
        roles=tuple(
            deployment.get("roles", ["sensor", "relay", "coordinator", "actuator"])
        ),
        output_dir=Path(outputs.get("directory", "outputs")),
        load_level=str(simulation.get("load_level", "medium")),
        min_neighbors=int(network.get("min_neighbors", 3)),
        max_neighbors=int(network.get("max_neighbors", 6)),
        link_drop_probability=float(network.get("link_drop_probability", 0.02)),
        link_add_probability=float(network.get("link_add_probability", 0.04)),
        initial_trust=float(social.get("initial_trust", 0.5)),
        trust_learning_rate=float(social.get("trust_learning_rate", 0.1)),
        social_weight_rho=float(social.get("social_weight_rho", 0.7)),
        privacy_threshold_min=float(privacy.get("threshold_min", 0.60)),
        privacy_threshold_max=float(privacy.get("threshold_max", 0.95)),
        min_energy=float(resources.get("min_energy", 0.05)),
        min_bandwidth=float(resources.get("min_bandwidth", 0.05)),
        min_compute=float(resources.get("min_compute", 0.05)),
        initial_energy_min=float(resources.get("initial_energy_min", 0.65)),
        initial_energy_max=float(resources.get("initial_energy_max", 1.0)),
        initial_bandwidth_min=float(resources.get("initial_bandwidth_min", 0.55)),
        initial_bandwidth_max=float(resources.get("initial_bandwidth_max", 1.0)),
        initial_compute_min=float(resources.get("initial_compute_min", 0.55)),
        initial_compute_max=float(resources.get("initial_compute_max", 1.0)),
        complexity_min=float(tasks.get("complexity_min", 0.2)),
        complexity_max=float(tasks.get("complexity_max", 1.0)),
        data_sensitivity_min=float(tasks.get("data_sensitivity_min", 0.0)),
        data_sensitivity_max=float(tasks.get("data_sensitivity_max", 1.0)),
        base_delay_ms=float(communication.get("base_delay_ms", 50.0)),
        utility_weight_system=float(utility_weights.get("system", 5.0 / 18.0)),
        utility_weight_social=float(utility_weights.get("social", 2.0 / 9.0)),
        utility_weight_resource=float(utility_weights.get("resource", 1.0 / 6.0)),
        utility_weight_privacy=float(utility_weights.get("privacy", 1.0 / 6.0)),
        utility_weight_incentive=float(utility_weights.get("incentive", 1.0 / 6.0)),
        utility_alpha_success=float(utility_system.get("alpha_success", 0.7)),
        utility_alpha_delay=float(utility_system.get("alpha_delay", 0.3)),
        utility_beta_trust=float(utility_social.get("beta_trust", 0.4)),
        utility_beta_preference=float(utility_social.get("beta_preference", 0.3)),
        utility_beta_reciprocity=float(utility_social.get("beta_reciprocity", 0.3)),
        utility_gamma_energy=float(utility_resource.get("gamma_energy", 0.4)),
        utility_gamma_bandwidth=float(utility_resource.get("gamma_bandwidth", 0.3)),
        utility_gamma_compute=float(utility_resource.get("gamma_compute", 0.3)),
        utility_delta_privacy=float(utility_privacy.get("delta_privacy", 1.0)),
        utility_eta_resource=float(utility_fairness.get("eta_resource", 0.5)),
        utility_eta_load=float(utility_fairness.get("eta_load", 0.5)),
        utility_zeta_coop=float(utility_incentive.get("zeta_coop", 0.4)),
        utility_zeta_efficiency=float(utility_incentive.get("zeta_efficiency", 0.3)),
        utility_zeta_trust=float(utility_incentive.get("zeta_trust", 0.3)),
    )


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a mapping section from parsed YAML data."""
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML using PyYAML when available, with a tiny fallback parser."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the simple nested mapping YAML used by the default config."""
    parsed: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, parsed)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        stripped = line.strip()
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                current[key] = _coerce_scalar(value)
            else:
                child: dict[str, Any] = {}
                current[key] = child
                stack.append((indent, child))
    return parsed


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip('"') for item in value[1:-1].split(",") if item]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"')

"""
evaluation_simulator.py

Defines evaluator behavior models (LLM, human, hybrid) for GenAI experimentation.

Each evaluator has parameters describing:
- Systematic bias (favoring or penalizing certain variants)
- Noise (consistency / rating variance)
- Cost per evaluation (e.g., token cost, human labor)
- Latency per evaluation (e.g., API vs manual review time)
- Confidence weight (for hybrid evaluators)
"""

# ---------------------------------------------------
# Evaluator data model
# ---------------------------------------------------

"""
Defines evaluator configurations (LLM, human, hybrid)
for the GenAI Experimentation Simulator.
Each evaluator has bias, noise, cost, latency, and description.
"""

from dataclasses import dataclass


@dataclass
class Evaluator:
    """Encapsulates one evaluator’s characteristics."""
    name: str
    bias: dict  # bias per variant, e.g. {"A": 0.1, "B": -0.05}
    noise: float  # random variance per evaluation
    cost_per_eval: float  # dollar cost per evaluation
    latency_per_eval: float  # seconds per evaluation
    description: str = ""


def list_evaluators():
    """
    Returns a dictionary of evaluator name → Evaluator instance.
    Each evaluator simulates different accuracy, cost, and bias profiles.
    """
    return {
        "LLM_v1": Evaluator(
            name="LLM_v1",
            bias={"A": 0.1, "B": -0.05},
            noise=0.3,
            cost_per_eval=0.002,
            latency_per_eval=0.6,
            description="Baseline LLM evaluator with moderate bias and noise. Cheap, fast, but imperfect."
        ),
        "LLM_v2": Evaluator(
            name="LLM_v2",
            bias={"A": 0.05, "B": 0.0},
            noise=0.2,
            cost_per_eval=0.004,
            latency_per_eval=0.7,
            description="Improved fine-tuned LLM with reduced noise and bias."
        ),
        "Human": Evaluator(
            name="Human",
            bias={"A": 0.0, "B": 0.0},
            noise=0.1,
            cost_per_eval=0.05,
            latency_per_eval=5.0,
            description="Human evaluators — high accuracy, unbiased, slow and costly."
        ),
        "Hybrid": Evaluator(
            name="Hybrid",
            bias={"A": 0.02, "B": -0.02},
            noise=0.15,
            cost_per_eval=0.015,
            latency_per_eval=1.5,
            description="Human-in-the-loop system combining LLM pre-screening with human verification."
        ),
    }


def get_evaluator(name: str) -> Evaluator:
    """Helper to fetch a single evaluator by name."""
    evaluators = list_evaluators()
    if name not in evaluators:
        raise ValueError(f"Unknown evaluator name: {name}")
    return evaluators[name]


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------

if __name__ == "__main__":

    print("Available evaluators:")
    evals = list_evaluators()
    for name, cfg in evals.items():
        print(name, cfg.bias, cfg.noise, cfg.description)


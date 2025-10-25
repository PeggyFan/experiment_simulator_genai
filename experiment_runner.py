"""
experiment_runner.py

Core experiment simulation logic for GenAI feature experimentation.

Simulates multiple variants (A, B, C...) with:
- Latent "true" performance (mean, variance)
- Evaluator bias (systematic offset)
- Evaluation noise (stochastic rating variability)
- Cost per evaluation (token or latency-based)
- Confidence intervals and PM-style decision recommendation
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def simulate_variant_scores(true_mean, true_std, n_samples, bias=0.0, eval_noise=0.3):
    """
    Simulate observed scores for a single variant.
    Combines true performance, evaluator bias, and random noise.
    """
    # True performance distribution
    true_scores = np.random.normal(true_mean, true_std, n_samples)
    # Add evaluator bias and scoring noise
    observed_scores = true_scores + bias + np.random.normal(0, eval_noise, n_samples)
    return observed_scores


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute mean and confidence interval for a sample.
    Returns mean, lower, and upper bounds.
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2.0, len(data) - 1)
    return mean, mean - h, mean + h


def estimate_cost(n_samples, cost_per_eval=0.002, latency_per_eval=0.5):
    """
    Estimate cost and total latency for given sample size.
    """
    total_cost = n_samples * cost_per_eval
    total_latency = n_samples * latency_per_eval
    return total_cost, total_latency


# ---------------------------------------------------
# Main runner
# ---------------------------------------------------

def run_experiment(
    variant_params,
    evaluator_bias,
    n_samples=1000,
    eval_noise=0.3,
    cost_per_eval=0.002,
    latency_per_eval=0.5,
    confidence=0.95,
):
    """
    Simulate an experiment across multiple GenAI feature variants.

    Parameters
    ----------
    variant_params : dict
        {"A": {"mean": 4.0, "std": 0.8}, "B": {"mean": 4.3, "std": 1.0}, ...}
    evaluator_bias : dict
        {"A": 0.0, "B": -0.2, ...}
    n_samples : int
        Number of evaluations per variant
    eval_noise : float
        Standard deviation of evaluator noise
    cost_per_eval : float
        Token or compute cost per evaluation
    latency_per_eval : float
        Latency per evaluation (seconds)
    confidence : float
        Confidence interval (default 95%)

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with variant metrics (mean, CI, cost, etc.)
    """

    results = []

    for variant, params in variant_params.items():
        # Simulate observed scores under evaluator
        scores = simulate_variant_scores(
            true_mean=params["mean"],
            true_std=params["std"],
            n_samples=n_samples,
            bias=evaluator_bias.get(variant, 0.0),
            eval_noise=eval_noise,
        )

        mean, ci_low, ci_high = compute_confidence_interval(scores, confidence)
        cost, latency = estimate_cost(n_samples, cost_per_eval, latency_per_eval)

        results.append({
            "Variant": variant,
            "TrueMean": params["mean"],
            "TrueStd": params["std"],
            "EvaluatorBias": evaluator_bias.get(variant, 0.0),
            "ObservedMean": mean,
            "CILow": ci_low,
            "CIHigh": ci_high,
            "Samples": n_samples,
            "TotalCost($)": cost,
            "TotalLatency(s)": latency,
        })

    results_df = pd.DataFrame(results)
    return results_df


# ---------------------------------------------------
# Example usage (run standalone)
# ---------------------------------------------------

if __name__ == "__main__":
    variant_params = {
        "A": {"mean": 4.0, "std": 0.8},
        "B": {"mean": 4.3, "std": 1.0},
    }

    evaluator_bias = {
        "A": 0.1,   # LLM evaluator slightly favors A
        "B": -0.2,  # Penalizes B's phrasing or tone
    }

    df = run_experiment(
        variant_params=variant_params,
        evaluator_bias=evaluator_bias,
        n_samples=500,
        eval_noise=0.3,
        cost_per_eval=0.002,
        latency_per_eval=0.6,
    )

    print("\n=== Experiment Results ===")
    print(df)

    best = df.loc[df["ObservedMean"].idxmax()]
    print(f"\nRecommended: Launch Variant {best['Variant']} (Observed Mean = {best['ObservedMean']:.2f})")

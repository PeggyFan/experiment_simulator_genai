"""
simulation_engine.py

Orchestrates GenAI feature experimentation simulations.

Responsibilities:
- Manage multiple evaluators (LLM, human, hybrid)
- Run repeated experiments (replications) per evaluator
- Aggregate outcomes, compute summary stats, and track cost/latency
- Provide structured results for dashboard visualization or PM reporting
"""

import pandas as pd
from experiment_runner import run_experiment
from evaluation_simulator import get_evaluator


# ---------------------------------------------------
# Core Simulation Function
# ---------------------------------------------------

def simulate_experiments(
    variant_params: dict,
    evaluator_names: list,
    n_runs: int = 20,
    n_samples: int = 500,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Run multiple simulated experiments across several evaluator types.

    Parameters
    ----------
    variant_params : dict
        Example: {"A": {"mean": 4.0, "std": 0.8}, "B": {"mean": 4.3, "std": 1.0}}
    evaluator_names : list[str]
        Evaluators to simulate, e.g. ["LLM_v1", "Human", "Hybrid"]
    n_runs : int
        Number of replicated experiments to run per evaluator
    n_samples : int
        Number of evaluations per variant per run
    confidence : float
        Confidence level for intervals

    Returns
    -------
    results_df : pd.DataFrame
        Aggregated simulation results across evaluators and runs
    """

    all_results = []

    for eval_name in evaluator_names:
        eval_cfg = get_evaluator(eval_name)
        # now eval_cfg is an Evaluator dataclass

        print(f"\n🔍 Running {n_runs} simulations for evaluator: {eval_cfg.name}")

        for run_id in range(1, n_runs + 1):
            df = run_experiment(
                variant_params=variant_params,
                evaluator_bias=eval_cfg.bias,
                n_samples=n_samples,
                eval_noise=eval_cfg.noise,
                cost_per_eval=eval_cfg.cost_per_eval,
                latency_per_eval=eval_cfg.latency_per_eval,
                confidence=confidence,
            )

            df["EvaluatorType"] = eval_cfg.name
            df["RunID"] = run_id
            all_results.append(df)

    results_df = pd.concat(all_results, ignore_index=True)
    return results_df


# ---------------------------------------------------
# Monte carlo Simulation for Decision Risk
# ---------------------------------------------------

def monte_carlo_decision_risk(
    variant_params,
    evaluator_names,
    n_runs=1000,
    n_samples=200,
):
    """
    Simulate many runs to estimate probability of wrong launch decisions.
    Returns: DataFrame with evaluator_name → wrong_decision_rate
    """
    from experiment_runner import run_experiment
    from evaluation_simulator import get_evaluator

    true_best_variant = max(variant_params.items(), key=lambda x: x[1]["mean"])[0]
    results = []

    for eval_name in evaluator_names:
        eval_cfg = get_evaluator(eval_name)
        wrong_decisions = 0

        for _ in range(n_runs):
            df = run_experiment(
                variant_params=variant_params,
                evaluator_bias=eval_cfg.bias,
                n_samples=n_samples,
                eval_noise=eval_cfg.noise,
                cost_per_eval=eval_cfg.cost_per_eval,
                latency_per_eval=eval_cfg.latency_per_eval,
            )

            observed_best = df.loc[df["ObservedMean"].idxmax(), "Variant"]
            if observed_best != true_best_variant:
                wrong_decisions += 1

        results.append({
            "EvaluatorType": eval_name,
            "WrongDecisionRate": wrong_decisions / n_runs,
            "TrueBestVariant": true_best_variant
        })

    return pd.DataFrame(results)


# ---------------------------------------------------
# Summary / Aggregation
# ---------------------------------------------------

def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across runs per evaluator and variant.
    Returns a summary DataFrame with mean observed scores and costs.
    """

    summary = (
        results_df
        .groupby(["EvaluatorType", "Variant"])
        .agg({
            "ObservedMean": ["mean", "std"],
            "TrueMean": "first",
            "EvaluatorBias": "first",
            "TotalCost($)": "mean",
            "TotalLatency(s)": "mean",
        })
        .reset_index()
    )

    # Flatten MultiIndex columns
    summary.columns = [
        "EvaluatorType",
        "Variant",
        "ObservedMean_Mean",
        "ObservedMean_Std",
        "TrueMean",
        "EvaluatorBias",
        "AvgCost($)",
        "AvgLatency(s)"
    ]
    return summary


# ---------------------------------------------------
# Decision Support: PM Summary Report
# ---------------------------------------------------

def generate_pm_summary(summary_df, reliability_df=None, mc_results=None):
    """
    Generate a natural-language PM-style summary of the simulation results.
    """
    lines = ["### 🧠 PM Summary: Experimentation Outcomes\n"]

    if mc_results is not None:
            text += "\n\n### Decision Risk Overview\n"
            for _, row in mc_results.iterrows():
                text += f"- {row['EvaluatorType']}: {row['WrongDecisionRate']:.2%} chance of wrong launch\n"
            text += "\nHigh wrong-decision rates indicate evaluator bias or noise leading to poor product calls."


    for eval_type in summary_df["EvaluatorType"].unique():
        sub = summary_df[summary_df["EvaluatorType"] == eval_type]
        best = sub.loc[sub["ObservedMean_Mean"].idxmax()]

        lines.append(f"**Evaluator:** {eval_type}")
        lines.append(
            f"- Best Variant: **{best['Variant']}** "
            f"(Observed Mean = {best['ObservedMean_Mean']:.2f}, True Mean = {best['TrueMean']:.2f})"
        )
        lines.append(
            f"- Cost per run ≈ ${best['AvgCost($)']:.3f}, "
            f"Avg latency ≈ {best['AvgLatency(s)']:.1f}s"
        )
        lines.append(
            f"- Evaluator bias → {best['EvaluatorBias']:+.2f} "
            f"({ 'systematic favor' if best['EvaluatorBias']>0 else 'penalty' if best['EvaluatorBias']<0 else 'neutral'})"
        )
        lines.append(
            f"- Interpretation: {'✅ Reliable signal, launch likely justified.' if abs(best['EvaluatorBias']) < 0.1 else '⚠️ Potential evaluator distortion — consider human calibration.'}"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------

if __name__ == "__main__":
    # print("Available evaluators:", list_evaluators())

    # Example setup
    variant_params = {
        "A": {"mean": 4.0, "std": 0.8},
        "B": {"mean": 4.3, "std": 1.0},
        "C": {"mean": 3.9, "std": 0.7},
    }

    evaluator_names = ["LLM_v1", "Human", "Hybrid"]

    results = simulate_experiments(
        variant_params=variant_params,
        evaluator_names=evaluator_names,
        n_runs=10,
        n_samples=400,
    )

    summary = summarize_results(results)
    print("\n=== Aggregated Summary ===")
    print(summary)

    print("\n=== PM Summary Report ===")
    print(generate_pm_summary(summary))

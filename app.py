"""
app.py

GenAI Experimentation Lab

A Streamlit app to simulate how biased, noisy, and costly evaluators (LLMs vs humans)
influence GenAI feature experiment outcomes.

Demonstrates PM tradeoffs between evaluation cost, accuracy, and decision reliability.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from evaluation_simulator import list_evaluators
from simulation_engine import simulate_experiments, summarize_results, generate_pm_summary

# ---------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="GenAI Experimentation Lab", layout="wide")
st.title("🧠 GenAI Experimentation Lab")

st.markdown("""
Simulate how **LLM-based evaluators** (vs human raters) can distort or accelerate A/B testing outcomes 
for GenAI product features — where “quality” is subjective and ground truth is uncertain.

This lab helps **AI PMs and operators** explore tradeoffs between *speed, cost, and reliability*.
""")

st.divider()

# ---------------------------------------------------
# SCENARIO SELECTOR
# ---------------------------------------------------

st.header("🧩 Choose an Experiment Scenario")

scenario = st.selectbox(
    "Select a simulation context:",
    [
        "Chatbot Helpfulness Evaluation",
        "Summarization Quality Comparison",
        "Prompt Engineering Variant Test",
        "Content Moderation Model Test",
        "AI Writing Assistant Tone Comparison",
    ],
)

# Scenario presets (can tweak)
scenario_defaults = {
    "Chatbot Helpfulness Evaluation": {"A": (4.0, 0.8), "B": (4.3, 1.0)},
    "Summarization Quality Comparison": {"A": (4.2, 0.6), "B": (4.4, 0.7)},
    "Prompt Engineering Variant Test": {"A": (4.1, 0.9), "B": (4.0, 0.7), "C": (4.3, 0.8)},
    "Content Moderation Model Test": {"A": (4.5, 0.5), "B": (4.2, 0.8)},
    "AI Writing Assistant Tone Comparison": {"A": (3.9, 1.0), "B": (4.3, 0.9)},
}

variant_params = {
    name: {"mean": m, "std": s}
    for name, (m, s) in scenario_defaults[scenario].items()
}

st.markdown(f"**Selected scenario:** {scenario}")
st.caption("Each variant represents a different LLM model, prompt template, or configuration being tested.")

st.divider()

# ---------------------------------------------------
# EVALUATOR PROFILES
# ---------------------------------------------------

st.header("👥 Evaluator Profiles")

evaluators_dict = list_evaluators()
eval_profiles = pd.DataFrame([
    {
        "Evaluator": name,
        "BiasDict": cfg.bias,  # keep original dict for display
        "Bias (Mean)": np.mean(list(cfg.bias.values())),
        "Bias (|Mean|)": np.mean(np.abs(list(cfg.bias.values()))),
        "Noise": cfg.noise,
        "Cost ($/eval)": cfg.cost_per_eval,
        "Latency (s)": cfg.latency_per_eval,
        "Description": cfg.description,
    }
    for name, cfg in evaluators_dict.items()
])

st.dataframe(eval_profiles, use_container_width=True)

# Bias vs Noise visualization
st.subheader("📉 Evaluator Bias–Noise Tradeoff")
fig, ax = plt.subplots(figsize=(4.5, 2.8))

ax.scatter(
    eval_profiles["Bias (Mean)"],
    eval_profiles["Noise"],
    s=200 * eval_profiles["Cost ($/eval)"],
    alpha=0.7,
)
for _, row in eval_profiles.iterrows():
    ax.text(row["Bias (Mean)"], row["Noise"], row["Evaluator"], fontsize=9)

ax.axvline(0, color="gray", linestyle="--")
ax.set_xlabel("Evaluator Bias (systematic offset)")
ax.set_ylabel("Noise (random variance)")
ax.set_title("Bias–Noise–Cost Landscape of Evaluators")
ax.tick_params(axis='both', labelsize=6)
ax.legend()
st.pyplot(fig, use_container_width=False)

# Evaluator selection
st.markdown("### Select Evaluators to Compare")
selected_evals = st.multiselect(
    "Choose evaluator types",
    list(evaluators_dict.keys()),
    default=["LLM_v1", "Human"]
)

st.divider()

# ---------------------------------------------------
# SIMULATION CONTROLS
# ---------------------------------------------------

st.header("⚙️ Simulation Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    n_samples = st.slider("Samples per Variant", 100, 2000, 500, step=100,
                          help="Each sample represents one LLM evaluation or human rating.")
with col2:
    n_runs = st.slider("Simulation Replications", 1, 50, 10,
                       help="How many independent simulated experiments to run.")
with col3:
    confidence = st.slider("Confidence Interval", 0.8, 0.99, 0.95, 0.01)

simulate_drift = st.checkbox("Include Evaluator Drift Over Time", value=False,
                             help="Simulate LLM model updates or prompt-induced drift.")
simulate_prompt_effect = st.checkbox("Include Prompt Sensitivity Noise", value=False,
                                     help="Adds random variance due to different prompt framings.")

run_sim = st.button("🚀 Run Simulation")

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
if run_sim:
    st.subheader("🧮 Running Simulation...")
    with st.spinner("Simulating evaluator behavior across multiple experiments..."):
        results_df = simulate_experiments(
            variant_params=variant_params,
            evaluator_names=selected_evals,
            n_runs=n_runs,
            n_samples=n_samples,
            confidence=confidence,
        )

        summary_df = summarize_results(results_df)
        pm_report = generate_pm_summary(summary_df)

    # ---------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------
    st.success("✅ Simulation Complete!")

    st.markdown("### 📈 Aggregated Summary of Observed Metrics")
    st.dataframe(summary_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
         # Cost vs accuracy tradeoff
        st.subheader("💸 Cost vs Observed Mean")
        fig3, ax3 = plt.subplots(figsize=(4.5, 2.8))
        for eval_type in summary_df["EvaluatorType"].unique():
            sub = summary_df[summary_df["EvaluatorType"] == eval_type]
            ax3.scatter(sub["AvgCost($)"], sub["ObservedMean_Mean"], label=eval_type)
        ax3.set_xlabel("Average Cost ($)")
        ax3.set_ylabel("Observed Mean Score")
        ax.tick_params(axis='both', labelsize=6)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig3, use_container_width=True)

    with col2:
         # Plot observed mean vs true mean
        st.subheader("🎯 Observed Mean vs True Mean (Evaluator Bias)")
        fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
        for eval_type in summary_df["EvaluatorType"].unique():
            sub = summary_df[summary_df["EvaluatorType"] == eval_type]
            ax2.bar(sub["Variant"] + f" ({eval_type})", sub["ObservedMean_Mean"], yerr=sub["ObservedMean_Std"], label=eval_type)
        ax2.axhline(y=list(variant_params.values())[0]["mean"], color="gray", linestyle="--", label="True Means")
        ax2.set_ylabel("Observed Mean Score")
        ax2.set_xlabel("Evaluator and Variant")
        ax.tick_params(axis='both', labelsize=4)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig2, use_container_width=True)


    # PM Summary
    st.markdown("---")
    st.markdown("## 🧠 PM Decision Summary")
    st.markdown(pm_report)

    st.download_button(
        label="💾 Download PM Summary Report",
        data=pm_report,
        file_name="pm_summary.md",
        mime="text/markdown"
    )

else:
    st.info("Adjust parameters and click **🚀 Run Simulation** to begin.")

# ---------------------------------------------------
# MONTE CARLO DECISION RISK VISUALIZATION
# ---------------------------------------------------
from simulation_engine import monte_carlo_decision_risk

st.subheader("🎲 Monte Carlo: Wrong Launch Decision Risk")

st.caption("""
Simulate **1000 experiments** to estimate how often each evaluator would 
launch the *wrong variant* (not the true best one).  
This captures the **decision instability** due to evaluator bias and noise.
""")

mc_button = st.button("🎲 Run Monte Carlo Simulation (1000 runs)")

if mc_button:
    with st.spinner("Running 1000 Monte Carlo experiments per evaluator..."):
        mc_results = monte_carlo_decision_risk(
            variant_params=variant_params,
            evaluator_names=selected_evals,
            n_runs=1000,
            n_samples=n_samples,
        )

    st.success("✅ Monte Carlo simulation complete!")
    st.dataframe(mc_results, use_container_width=True)

    # Plot distribution of wrong decision rates
    fig_mc, ax_mc = plt.subplots(figsize=(4.5, 2.8))
    ax_mc.bar(mc_results["EvaluatorType"], mc_results["WrongDecisionRate"], color="tomato")
    ax_mc.set_ylabel("Wrong Decision Rate")
    ax_mc.set_ylim(0, 1)
    ax_mc.set_title("Probability of Launching Wrong Variant")
    ax.tick_params(axis='both', labelsize=4)
    for i, v in enumerate(mc_results["WrongDecisionRate"]):
        ax_mc.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    st.pyplot(fig_mc, use_container_width=False)

    st.markdown(f"🧠 **True best variant:** `{mc_results.iloc[0]['TrueBestVariant']}`")
    st.markdown("""
    Lower bars = more reliable evaluator decisions.
    High wrong-decision rates mean the evaluator's bias or noise 
    can lead to false launches, increasing risk for PMs.
    """)


st.divider()

# ---------------------------------------------------
# EDUCATIONAL SECTION
# ---------------------------------------------------
with st.expander("💡 Under the Hood: Why LLM Experimentation is Different"):
    st.markdown("""
    **Traditional A/B testing** assumes deterministic metrics and clear ground truth (like CTR or retention).  
    **GenAI features** break these assumptions:
    - LLM outputs are *stochastic* — same prompt, different answers.
    - “Quality” is subjective and often judged by other models.
    - LLM evaluators can have *systematic biases* (e.g., preferring verbosity).
    - Each evaluation has a *real cost* (tokens, latency, API time).
    - The “true best variant” is uncertain, even post-launch.

    This simulator helps PMs quantify when LLM-based evaluation is *good enough* to launch — 
    and when to bring humans back in the loop.
    """)

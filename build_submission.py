"""Create a draft report and whitelist-based upload ZIP from saved evidence only.

Report-only dependencies: requirements-report.txt. No project training is run.
"""
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
from xml.sax.saxutils import escape
import csv
import hashlib
import json
import re
import shutil
import zipfile
import os

ROOT = Path(__file__).resolve().parent
NAME = "CS664_Project_Wong_Sheung_Wan"
RESULTS = ROOT / "results"
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import numpy as np
import nbformat
from scipy.stats import t as student_t


def rows(name):
    with (RESULTS / name).open(newline="") as file:
        return list(csv.DictReader(file))


def money(value, places=2):
    return f"${float(value):,.{places}f}"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_saved_evidence():
    main = nbformat.read(ROOT / "CS664_Flight_Booking.ipynb", as_version=4)
    code = [cell for cell in main.cells if cell.cell_type == "code"]
    assert len(code) == 80
    assert [cell.execution_count for cell in code] == list(range(1, 81))
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    summary = json.loads((RESULTS / "run_summary.json").read_text())
    assert {k: summary[k] for k in (
        'training_seed', 'test_seed', 'training_episodes', 'test_trajectories',
        'alpha', 'gamma', 'epsilon_start', 'epsilon_end', 'epsilon_decay_episodes',
        'evaluation_exploration', 'official_threshold')} == dict(
        training_seed=66401, test_seed=66402, training_episodes=50000,
        test_trajectories=10000, alpha=.1, gamma=1., epsilon_start=1.,
        epsilon_end=.05, epsilon_decay_episodes=40000, evaluation_exploration=0,
        official_threshold=180)
    assumptions = json.loads((RESULTS / 'simulation_assumptions.json').read_text())
    assert assumptions == summary['simulation_assumptions']
    assert assumptions['price_levels'] == [120, 180, 240, 300, 360]
    assert assumptions['time_remaining'] == list(range(6, -1, -1))
    assert assumptions['early_wait_times'] == [6, 5, 4, 3]
    assert assumptions['late_wait_times'] == [2, 1]
    assert assumptions['early_probabilities'] == [.3, .3, .4]
    assert assumptions['late_probabilities'] == [.6, .2, .2]
    budget_config = json.loads((RESULTS / 'supplementary_experiment_config.json').read_text())
    dynamics_config = json.loads((RESULTS / 'learning_dynamics_config.json').read_text())
    assert budget_config['training_seeds'] == dynamics_config['seeds'] == [66401, 66411, 66421, 66431, 66441]
    assert budget_config['episode_budgets'] == [50000, 200000]
    assert budget_config['alpha'] == .1 and budget_config['gamma'] == 1
    assert dynamics_config['episodes_per_run'] == 200000 and dynamics_config['instrumented_runs'] == 15
    assert dynamics_config['decay_scale'] == 1000
    assert [.1 / (1 + n / 1000) for n in [0, 1000, 3000]] == [.1, .05, .025]
    paired = rows("paired_purchase_costs.csv")
    assert len(paired) == 10_000
    q = np.array([int(row["Q-learning (greedy)"]) for row in paired])
    for row in rows("mean_cost_comparison.csv"):
        costs = np.array([int(r[row["strategy"]]) for r in paired])
        assert costs.mean() == float(row["mean_purchase_cost"])
    for row in rows("paired_savings_95ci.csv"):
        diff = np.array([int(r[row["baseline"]]) for r in paired]) - q
        half = student_t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
        assert np.isclose(float(row["mean_paired_saving"]), diff.mean(), rtol=0, atol=1e-10)
        assert np.isclose(float(row["ci_95_lower"]), diff.mean() - half, rtol=0, atol=1e-10)
        assert np.isclose(float(row["ci_95_upper"]), diff.mean() + half, rtol=0, atol=1e-10)
    exact = summary["exact_mdp"]
    assert np.isclose(exact["gap_above_optimal"], exact["learned_policy_expected_cost"] - exact["optimal_expected_cost"])
    states = rows("exact_state_values.csv")
    starts = [r for r in states if r["time_remaining"] == "6"]
    assert np.isclose(np.mean([float(r["optimal_expected_cost"]) for r in starts]), exact["optimal_expected_cost"])
    assert np.isclose(np.mean([float(r["learned_policy_expected_cost"]) for r in starts]), exact["learned_policy_expected_cost"])
    update = summary["first_update"]
    assert update["terminal"] and update["target"] == update["reward"]
    assert np.isclose(update["updated"], update["old"] + update["learning_rate"] * (update["target"] - update["old"]))
    trace = rows("q_learning_trace.csv")
    assert trace[-1]["action"] == "BOOK" and all(r["action"] == "WAIT" for r in trace[:-1])
    assert sum(int(r["reward"]) for r in trace) == -summary["final_purchase_cost"]
    budget_runs = rows("supplementary_training_runs.csv")
    budget_pairs = rows("supplementary_paired_gap_changes.csv")
    assert len(budget_runs) == 10 and len(budget_pairs) == 5
    for row in budget_pairs:
        assert np.isclose(float(row["gap_reduction_50000_minus_200000"]), float(row["gap_50000"]) - float(row["gap_200000"]))
        assert float(row["gap_reduction_50000_minus_200000"]) < 0
    for row in rows("supplementary_budget_summary.csv"):
        values = [float(r["exact_expected_cost"]) for r in budget_runs if r["episodes"] == row["episodes"]]
        assert np.isclose(float(row["mean_exact_cost"]), np.mean(values))
        assert float(row["min_exact_cost"]) == min(values) and float(row["max_exact_cost"]) == max(values)
    finals = rows("learning_dynamics_final_comparison.csv")
    assert len(finals) == 15 and len(rows("learning_dynamics_checkpoints.csv")) == 300
    assert len(rows("learning_dynamics_tracked_states.csv")) == 2100
    for row in rows("learning_dynamics_paired_comparisons.csv"):
        assert np.isclose(float(row["reduction_0.1_minus_0.03"]), float(row["cost_alpha_0.1"]) - float(row["cost_alpha_0.03"]))
        assert np.isclose(float(row["reduction_0.1_minus_decay"]), float(row["cost_alpha_0.1"]) - float(row["cost_visit_decay"]))
        assert float(row["reduction_0.1_minus_0.03"]) > 0 and float(row["reduction_0.1_minus_decay"]) > 0
    for row in rows("learning_dynamics_across_seed_summary.csv"):
        values = [float(r["exact_expected_cost"]) for r in finals if r["condition"] == row["condition"]]
        assert np.isclose(float(row["mean_exact_cost"]), np.mean(values))
        assert float(row["min_exact_cost"]) == min(values) and float(row["max_exact_cost"]) == max(values)
        assert np.isclose(float(row['mean_gap']), np.mean(values) - exact['optimal_expected_cost'])
    assert all(float(row["gap_from_optimal"]) > 0 for row in finals)
    baselines = rows('supplementary_baseline_comparisons.csv')
    for baseline, count in [('Immediate booking', 10), ('Minimum-price rule ($120)', 10),
                            ('Fixed-threshold rule ($180)', 7)]:
        selected = [r for r in baselines if r['baseline'] == baseline]
        assert sum(float(r['learned_exact_cost']) < float(r['baseline_exact_cost']) for r in selected) == count
    # Audit numerical examples stated in prose as well as the generated tables.
    suboptimal = rows("supplementary_suboptimal_states.csv")
    assert len(suboptimal) == 7
    first = next(r for r in suboptimal if r["time_remaining"] == "6" and r["price"] == "180")
    assert (int(first["book_visits"]), int(first["wait_visits"])) == (5469, 4490)
    assert np.isclose(float(first["exact_action_penalty"]), 16.2168)
    assert all(r["has_lower_quartile_legal_action"] == "False" for r in suboptimal)
    supplementary = json.loads((RESULTS / "supplementary_summary.json").read_text())
    assert supplementary["lower_quartile_visit_cutoff"] == 326
    assert summary["verification"]["legal_pairs_visited"] == 65
    observed = summary["observed_wait_update"]
    assert observed["episode"] == 376 and observed["state"] == [1, 240]
    assert (observed["old"], observed["reward"], observed["target"], observed["updated"]) == (0, 0, -30, -3)
    assert np.isclose(observed["updated"], observed["old"] + observed["learning_rate"] * (observed["target"] - observed["old"]))
    conditions = ["constant_0.1", "constant_0.03", "visit_decay"]
    for condition, counts, mean_gap, wait_count in zip(
            conditions, [(39, 48, 8), (46, 40, 9), (36, 26, 33)],
            [13.704336, 6.0506592, 1.890720], [0, 2, 5]):
        change = next(r for r in rows("learning_dynamics_change_summary.csv") if r["condition"] == condition)
        assert tuple(int(change[k]) for k in ["improved_intervals", "worsened_intervals", "unchanged_intervals"]) == counts
        group = [r for r in finals if r["condition"] == condition]
        assert np.isclose(np.mean([float(r["gap_from_optimal"]) for r in group]), mean_gap)
        state = next(r for r in rows("learning_dynamics_state_summary.csv")
                     if r["condition"] == condition and r["time_remaining"] == "6" and r["price"] == "180")
        assert int(state["seeds_wait"]) == wait_count
    change = next(r for r in rows("learning_dynamics_checkpoint_changes.csv")
                  if r["training_seed"] == "66401" and r["condition"] == "constant_0.1"
                  and r["from_episode"] == "20000" and r["to_episode"] == "30000")
    assert np.isclose(float(change["exact_cost_change"]), 4.719312)
    assert int(change["tracked_state_action_switches"]) == 4
    assert change["tracked_switch_details"].count("WAIT->BOOK") == 4
    for filename in ["verification_results.csv", "supplementary_verification.csv", "learning_dynamics_verification.csv"]:
        assert all(row["status"] == "PASS" for row in rows(filename))
    start = datetime.fromisoformat(code[0].metadata["execution"]["iopub.status.busy"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(code[-1].metadata["execution"]["iopub.status.idle"].replace("Z", "+00:00"))
    runtime = (end - start).total_seconds()
    audit = {"status": "PASS", "saved_executed_cells": len(code), "training_rerun": False,
             "recorded_cell_execution_seconds": runtime, "timing_excludes": "kernel startup and runner postchecks",
             "verified": ["10000 paired costs and means", "three paired Student-t intervals", "exact costs/gap",
                          "actual logged update", "complete trace", "ten budget runs", "fifteen rate runs",
                          "300 checkpoints/2100 diagnostics", "reported means/ranges", "saved PASS records",
                          "prose coverage, rate, action-switch, and update examples"]}
    (ASSETS / "report_number_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return summary, runtime


def make_report_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    checkpoints = rows("learning_dynamics_checkpoints.csv")
    conditions = ["constant_0.1", "constant_0.03", "visit_decay"]
    seeds = [66401, 66411, 66421, 66431, 66441]
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True, sharey=True, layout="constrained")
    for ax, condition in zip(axes, conditions):
        for seed in seeds:
            selected = [r for r in checkpoints if r["condition"] == condition and int(r["training_seed"]) == seed]
            ax.plot([int(r["completed_episodes"]) / 1000 for r in selected],
                    [float(r["gap_from_optimal"]) for r in selected], marker=".", linewidth=1, label=str(seed))
        ax.set_title(condition.replace("constant_", "Constant alpha = ").replace("visit_decay", "Visit-based decay"), fontsize=10)
        ax.set_ylabel("Exact gap ($)", fontsize=9)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=9)
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Completed episodes (thousands)", fontsize=10)
    axes[-1].set_xticks([10, 50, 100, 150, 200])
    axes[-1].set_ylim(-.5, max(float(r["gap_from_optimal"]) for r in checkpoints) + .5)
    target = ASSETS / "learning_dynamics_report.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def report_blocks(summary, runtime=None, graph=None):
    """Concise report from saved evidence; full run listings remain in results/."""
    blocks = []
    def title(text): blocks.append(("title", text))
    def h(text): blocks.append(("heading", text))
    def p(text): blocks.append(("paragraph", text))
    def table(headers, data, widths=None): blocks.append(("table", (headers, data, widths)))
    def page(): blocks.append(("pagebreak", None))
    exact = summary["exact_mdp"]
    title("Flight Booking Decision Assistant")
    p("CS664 Term Project | Prepared for Sheung Wan Wong | October 14, 2026")
    p("DRAFT — awaiting student review.")
    h("1. Project goal and simulator")
    p("This Python tabular Q-learning application chooses BOOK or WAIT for a simulated traveler with a fixed departure date. It tests whether current fare and time remaining can reduce expected purchase cost relative to simple booking rules. The proposal specified a seven-opportunity simulator, 50,000 training episodes, four strategies on 10,000 shared test paths, paired uncertainty estimates, an exact MDP reference, a policy heatmap, a decision trace, and verification [1, 2]. Supplementary experiments examine training budget and learning rate separately from the original comparison.")
    p("State is (time remaining, current price). Seven daily purchase opportunities run from time 6 to 0; fares are $120, $180, $240, $300, and $360. The starting fare is uniform over these five levels. WAIT uses the probability regime for the time before the action:")
    table(["Time before WAIT", "+$60", "Unchanged", "−$60"], [["6–3", "30%", "30%", "40%"], ["2–1", "60%", "20%", "20%"]])
    p("WAIT gives reward 0 and reduces time by one. Moves beyond $120–$360 stay at the boundary. BOOK gives the negative current fare as reward and terminates. At time 0, BOOK is mandatory and a requested WAIT becomes BOOK. Actions after purchase raise an error until reset. Refunds, cancellations, and sold-out events are excluded. All probabilities are hypothetical.")
    h("2. Training and an observed Q-value update")
    p("The agent learns from observed transitions without receiving their probability table [3]. Maximizing undiscounted negative purchase reward minimizes cost. Its zero-initialized Q-table has shape 7 × 5 × 2, with 65 legal state/action pairs; deadline WAIT is excluded and greedy ties choose BOOK.")
    table(["Original setting", "Value"], [["Episodes / learning rate / gamma", "50,000 / 0.1 / 1.0"], ["Training / test seeds", "66401 / 66402"], ["Epsilon schedule", "1.0 on episode 1 to 0.05 on episode 40,000; then 0.05"], ["Evaluation", "Greedy; epsilon 0; no Q updates"]], [2.5, 4.7])
    p("Q_new = Q_old + alpha × (target − Q_old). Terminal BOOK uses target = reward; otherwise target = reward + gamma × max Q(next state, legal action). Only BOOK is legal in a deadline next state.")
    u = summary['first_update']
    p(f"Actual logged example: episode {u['episode']}, state ({u['state'][0]}, ${u['state'][1]}), {u['action']}. Old Q = {u['old']:g}, reward and terminal target = {u['target']:g}, alpha = {u['learning_rate']:g}. Thus Q_new = 0 + 0.1 × (−120 − 0) = {u['updated']:g}. The saved update is in [run_summary.json](results/run_summary.json).")
    page()
    h("3. Original sampled four-strategy comparison")
    p("All four strategies used the same 10,000 fresh seven-price paths, generated with test seed 66402 independently of training seed 66401 [5]. Only the evaluator sees complete paths: each policy receives its current state, stops at purchase, and must book at the deadline. Evaluation has no exploration or Q updates. The $180 threshold was specified from analytical model knowledge in the proposal, without searching the test paths.")
    table(["Strategy", "Rule", "Sample mean cost"], [[r['strategy'], rule, money(r['mean_purchase_cost'])] for r, rule in zip(rows('mean_cost_comparison.csv'), ["BOOK first day", "BOOK at $120; else deadline", "BOOK at $180 or below; else deadline", "Original greedy policy"])], [2.3, 3.4, 1.5])
    table(["Baseline", "Mean paired saving", "Individual paired 95% CI"], [[r['baseline'], money(r['mean_paired_saving']), f"[{money(r['ci_95_lower'])}, {money(r['ci_95_upper'])}]" ] for r in rows('paired_savings_95ci.csv')], [2.7, 1.7, 2.8])
    p("Saving is baseline cost minus Q-learning cost on the same path. Each interval is mean(d) ± t(0.975, 9999) × sample_sd(d)/sqrt(10000), using SciPy's Student-t quantile [4]. All three intervals lie above zero. They estimate mean savings conditional on this trained policy; they are neither simultaneous intervals nor ranges for individual-trip savings. Training-seed variation is assessed separately below.")
    p("Evidence: [four-strategy means](results/mean_cost_comparison.csv), [paired savings and intervals](results/paired_savings_95ci.csv), and [all 10,000 paired purchase costs](results/paired_purchase_costs.csv). Monetary summaries are rounded to cents only after calculations from full-precision values; saved artifacts retain their precision.")
    h("4. Exact model expectations")
    p("Backward induction uses V*(0,p) = p and V*(t,p) = min[p, sum P(p′|t,p)V*(t−1,p′)]. Fixed-policy evaluation follows the chosen action instead of minimizing. Averaging over the five uniform starting fares gives full-model expectations, distinct from sampled means and estimated Q-values.")
    table(["Analytical reference", "Expected cost"], [["Optimal finite-horizon policy", money(exact['optimal_expected_cost'], 6)], ["Original learned policy", money(exact['learned_policy_expected_cost'], 6)], ["Learned minus optimal", money(exact['learned_policy_expected_cost'] - exact['optimal_expected_cost'], 6)]], [4.5, 2.7])
    p("The original policy beats the simple rules in the sample but remains suboptimal. At (6, $180), for example, it BOOKs while the optimal action is WAIT. Its sampled mean of $214.85 differs from its exact expectation of $215.709120 because the test sample is finite. See [exact policy costs](results/exact_mdp_comparison.csv) and [all state values](results/exact_state_values.csv).")
    page()
    h("5. Original learned policy and complete decision trace")
    blocks.append(("figure", (str(RESULTS / 'policy_heatmap.png'), "Figure 1. Original greedy policy: BOOK or WAIT at each state. Time 0 is mandatory BOOK.", 6.4, 4.8)))
    p("The trace is trajectory index 1, the first test example with at least three decisions. It shows every observed decision through purchase; unseen post-purchase prices are excluded.")
    table(["Day", "Time remaining", "Price", "Action", "Reward"], [[r['day'], r['time_remaining'], money(r['price'], 0), r['action'], r['reward']] for r in rows('q_learning_trace.csv')])
    p(f"The episode ends after {len(rows('q_learning_trace.csv'))} decisions with a {money(summary['final_purchase_cost'], 0)} purchase. It illustrates policy behavior, not average performance: positive mean savings can coexist with expensive trips. Evidence: [complete trace](results/q_learning_trace.csv) and [all policy actions](results/learned_policy_actions.csv).")
    page()
    h("6. Executed verification")
    table(["Core check", "Cases", "Outcome"], [[r['check'], f"{int(r['cases']):,}", r['status']] for r in rows('verification_results.csv')], [4.8, 1.0, 1.4])
    p("The exact reference was independently checked by enumerating 3,645 latent-move paths for each of five policies (18,225 comparisons). Supplementary checks passed for nonzero-old-value Q updates, deadline selection, all-state rewards and invalid actions, tie-aware penalties, independent training, and preservation of the original model. Equal-cost optimal actions are treated as ties.")
    p("Instrumentation verified update counters, applied learning rates, terminal/legal targets, and checkpoint isolation. All 300 checkpoints preserved Q, counts, and training RNG state. Constant alpha 0.1 exactly reproduced the ten earlier 50,000/200,000 endpoint Q-tables, visit counts, and analytical costs. The saved main Notebook has 80 successfully executed code cells. These finite checks support implementation behavior; they do not prove convergence.")
    p("Complete executed records: [core checks](results/verification_results.csv), [supplementary checks](results/supplementary_verification.csv), and [learning-dynamics checks](results/learning_dynamics_verification.csv).")
    h("7. Supplementary training-budget experiment")
    p("Seeds 66401, 66411, 66421, 66431, and 66441 each restarted from zero at 50,000 and 200,000 episodes (ten runs). Only episode budget changed: simulator, alpha 0.1, gamma 1, initialization, and the original epsilon schedule stayed fixed. Final greedy policies were evaluated exactly under the uniform initial-fare distribution.")
    table(["Episodes", "Mean exact cost", "Across-seed cost range"], [[f"{int(r['episodes']):,}", money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]" ] for r in rows('supplementary_budget_summary.csv')], [1.5, 2.2, 3.5])
    pairs = rows('supplementary_paired_gap_changes.csv')
    example = next(r for r in pairs if r['training_seed'] == '66401')
    increase = float(example['gap_200000']) - float(example['gap_50000'])
    p(f"All five seeds worsened at 200,000 episodes; seed 66401 increased by {money(increase)}. All ten models cost less than the immediate and $120 baselines in exact evaluation; seven cost less than the $180 baseline. The original policy had seven strictly suboptimal states, although all 65 legal pairs were visited and none of those states met the descriptive lower-quartile coverage flag. Coverage alone does not explain the optimality gap.")
    p("Full evidence: [all ten runs](results/supplementary_training_runs.csv), [paired budget changes](results/supplementary_paired_gap_changes.csv), [coverage diagnostics](results/supplementary_suboptimal_states.csv), and the [executed Notebook](CS664_Flight_Booking.ipynb).")
    page()
    h("8. Supplementary learning-rate experiment")
    p("The same five seeds independently restarted for 200,000 episodes under three prespecified conditions: constant alpha 0.1; constant alpha 0.03; and alpha(s,a) = 0.1/(1 + N_previous(s,a)/1000). N_previous counts updates to that pair before the current update. At 0, 1,000, and 3,000 prior updates, alpha is 0.1, 0.05, and 0.025. Scale 1,000 was chosen before the experiment. All other settings and the legal greedy evaluator stayed fixed.")
    p("Fifteen runs recorded 300 checkpoints and 2,100 diagnostics for the seven originally suboptimal states. Equal seeds do not ensure identical experience across conditions: changed policies can alter later transitions and RNG consumption. Every run, including equal-cost policies, remains in the evidence.")
    labels = {'constant_0.1': 'Constant 0.1', 'constant_0.03': 'Constant 0.03', 'visit_decay': 'Visit decay'}
    rate_summary = rows('learning_dynamics_across_seed_summary.csv')
    table(["Condition", "Mean exact cost", "Across-seed cost range", "Mean optimality gap"], [[labels[r['condition']], money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]", money(r['mean_gap'])] for r in rate_summary], [1.5, 1.5, 2.7, 1.5])
    comparison = next(r for r in rows('learning_dynamics_paired_comparisons.csv') if r['training_seed'] == '66401')
    reduction03 = float(comparison['cost_alpha_0.1']) - float(comparison['cost_alpha_0.03'])
    reduction_decay = float(comparison['cost_alpha_0.1']) - float(comparison['cost_visit_decay'])
    p(f"Both alternatives lowered every seed's final exact cost relative to alpha 0.1. For seed 66401, reductions were {money(reduction03)} with alpha 0.03 and {money(reduction_decay)} with visit decay. Every final policy still had a positive optimality gap. Checkpoint costs fluctuated in all conditions; action switches accompanied performance changes, but sparse checkpoints do not isolate their cause. No best-checkpoint selection was performed.")
    p("Complete evidence: [all fifteen final runs](results/learning_dynamics_final_comparison.csv), [paired rate comparisons](results/learning_dynamics_paired_comparisons.csv), [all checkpoints](results/learning_dynamics_checkpoints.csv), [state diagnostics](results/learning_dynamics_tracked_states.csv), and the [saved dynamics figure](results/learning_dynamics_checkpoint_gaps.png). Methods and outputs are also in the [main Notebook](CS664_Flight_Booking.ipynb).")
    h("9. Limitations")
    p("The model uses hypothetical probabilities, five fare levels, a short horizon, and no sold-out events; it provides no evidence of real-flight savings. Paired intervals quantify test-path uncertainty for one policy, not training uncertainty or individual-trip outcomes. Across-seed ranges describe five selected seeds, not confidence intervals; two budgets do not establish a continuous trend. Supplementary analyses are exploratory, not independent confirmation, and do not establish universal robustness or convergence. Fixed-step target variability, decay, learning speed, and coverage may matter, but their causal contributions were not isolated. No simultaneous inference or additional tuning is claimed.")
    h("10. Reproducibility and references")
    p("Follow [README setup and execution instructions](README.md) for the pinned environment, project interpreter, Notebook kernel, and fresh-kernel runner. [RESULTS.md](RESULTS.md) indexes the complete saved outputs. Full per-seed results and full-precision artifacts are retained; this report revision uses saved evidence without rerunning experiments.")
    for text in [
        "[1] Kalathur, S. (2026). CS664 Term Project. Instructor-provided metcs664_TermProject.pdf; official project requirements.",
        "[2] [Original project proposal](CS664_Project_Wong_Sheung_Wan_Proposal.md); [executed Notebook](CS664_Flight_Booking.ipynb). Numerical audit: [report_number_audit.json](report_assets/report_number_audit.json).",
        "[3] Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292. [Publisher reference](https://doi.org/10.1007/BF00992698). Algorithm background; no convergence theorem is claimed for this experiment.",
        "[4] SciPy contributors. [scipy.stats.t documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html). Student-t quantile; consulted October 14, 2026.",
        "[5] NumPy contributors. [Random Generator documentation](https://numpy.org/doc/stable/reference/random/generator.html). Seeded random streams; consulted October 14, 2026."]:
        p(text)
    return blocks


def make_documents(blocks):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.units import inch
    import matplotlib
    def pdf_text(text):
        # Keep evidence links usable in all three report formats.
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                      lambda m: '<link href="' + escape(m[2], {'"': '&quot;'}) +
                      '" color="#184e77"><u>' + m[1] + '</u></link>', escape(text))
    font_folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("Report", str(font_folder / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ReportBold", str(font_folder / "DejaVuSans-Bold.ttf")))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyReport", fontName="Report", fontSize=9.5, leading=13.2, spaceAfter=7))
    styles.add(ParagraphStyle("HeadingReport", fontName="ReportBold", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=6, keepWithNext=True, textColor=colors.HexColor("#184e77")))
    styles.add(ParagraphStyle("TitleReport", fontName="ReportBold", fontSize=20, leading=24, spaceAfter=12))
    styles.add(ParagraphStyle("TableReport", fontName="Report", fontSize=8.3, leading=11))
    styles.add(ParagraphStyle("TableHeading", fontName="ReportBold", fontSize=8.3, leading=11, textColor=colors.white))
    styles.add(ParagraphStyle("CaptionReport", fontName="Report", fontSize=8.6, leading=12, spaceAfter=9))
    flow=[]
    for kind, data in blocks:
        if kind in ("title", "heading", "paragraph"):
            key={"title":"TitleReport","heading":"HeadingReport","paragraph":"BodyReport"}[kind]
            flow.append(Paragraph(pdf_text(data),styles[key]))
        elif kind=="table":
            headers,values,widths=data
            widths=[w*inch for w in widths] if widths else [7.2*inch/len(headers)]*len(headers)
            body=[[Paragraph(escape(str(v)),styles["TableHeading" if i==0 else "TableReport"]) for v in row]
                  for i,row in enumerate([headers]+values)]
            table=Table(body,colWidths=widths,repeatRows=1,hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#184e77")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f5f9")]),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),
                ("BOTTOMPADDING",(0,0),(-1,-1),5),("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#184e77"))]))
            flow.extend([table,Spacer(1,8)])
        elif kind=="figure":
            path,caption,width,height=data
            flow.extend([Image(path,width=width*inch,height=height*inch),Paragraph(escape(caption),styles["CaptionReport"])])
        elif kind=="pagebreak":flow.append(PageBreak())
    def footer(canvas,doc):
        canvas.setFont("Report",8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(.65*inch,.35*inch,"DRAFT — awaiting student review")
        canvas.drawRightString(7.85*inch,.35*inch,f"Page {doc.page}")
    pdf=ROOT/f"{NAME}.pdf"
    SimpleDocTemplate(str(pdf),pagesize=(8.5*inch,11*inch),leftMargin=.65*inch,rightMargin=.65*inch,
        topMargin=.6*inch,bottomMargin=.65*inch,title="Flight Booking Decision Assistant — Draft",author="Prepared for Sheung Wan Wong").build(flow,onFirstPage=footer,onLaterPages=footer)

    from docx import Document
    from docx.shared import Inches,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    document=Document()
    section=document.sections[0]
    section.page_width,section.page_height=Inches(8.5),Inches(11)
    section.left_margin=section.right_margin=Inches(.65)
    section.top_margin=section.bottom_margin=Inches(.6)
    normal=document.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(6)
    document.core_properties.title="Flight Booking Decision Assistant — Draft"
    document.core_properties.author="Prepared for Sheung Wan Wong"
    def add_linked_paragraph(text):
        from docx.opc.constants import RELATIONSHIP_TYPE as RT
        paragraph = document.add_paragraph()
        end = 0
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
            paragraph.add_run(text[end:match.start()])
            link = OxmlElement('w:hyperlink')
            link.set(qn('r:id'), paragraph.part.relate_to(match[2], RT.HYPERLINK, is_external=True))
            run = OxmlElement('w:r')
            props = OxmlElement('w:rPr')
            color = OxmlElement('w:color'); color.set(qn('w:val'), '184E77'); props.append(color)
            underline = OxmlElement('w:u'); underline.set(qn('w:val'), 'single'); props.append(underline)
            run.append(props)
            label = OxmlElement('w:t'); label.text = match[1]; run.append(label)
            link.append(run); paragraph._p.append(link)
            end = match.end()
        paragraph.add_run(text[end:])
    for kind,data in blocks:
        if kind=='title':document.add_heading(data,0)
        elif kind=='heading':document.add_heading(data,1)
        elif kind=='paragraph':add_linked_paragraph(data)
        elif kind=='pagebreak':document.add_page_break()
        elif kind=='figure':
            path,caption,width,height=data
            paragraph=document.add_paragraph();paragraph.paragraph_format.keep_with_next=True
            paragraph.add_run().add_picture(path,width=Inches(width),height=Inches(height))
            document.add_paragraph(caption,style='Caption')
        elif kind=='table':
            headers,values,widths=data
            table=document.add_table(rows=1,cols=len(headers));table.style='Light Shading Accent 1'
            for cell,text in zip(table.rows[0].cells,headers):cell.text=str(text)
            repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for row in values:
                for cell,value in zip(table.add_row().cells,row):cell.text=str(value)
            table.autofit=False
            for row in table.rows:
                no_split=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(no_split)
                for i,cell in enumerate(row.cells):
                    cell.width=Inches(widths[i] if widths else 7.2/len(headers))
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after=Pt(3)
                        for run in paragraph.runs:run.font.size=Pt(9)
            document.add_paragraph()
    paragraph=section.footer.paragraphs[0]
    paragraph.text='DRAFT — awaiting student review | Page '
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');paragraph._p.append(field)
    document.save(ROOT/f'{NAME}.docx')
    # A readable shared source for future human editing/rebuilding.
    markdown=[]
    for kind,data in blocks:
        if kind in ('title','heading','paragraph'):
            markdown.append(('# ' if kind=='title' else '## ' if kind=='heading' else '')+data)
        elif kind=='table':
            headers,values,_=data
            markdown.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+
                '\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in values))
        elif kind=='figure':markdown.append(f"![{data[1]}]({Path(data[0]).relative_to(ROOT)})")
    (ROOT/'REPORT_DRAFT.md').write_text('\n\n'.join(markdown)+'\n')


def verify_pdf_and_docx():
    import pymupdf as fitz
    from docx import Document
    from PIL import Image,ImageOps,ImageDraw
    folder=ROOT/'.cache'/'report_preview';folder.mkdir(parents=True,exist_ok=True)
    pdf=fitz.open(ROOT/f'{NAME}.pdf')
    text='\n'.join(page.get_text() for page in pdf)
    assert 5 <= len(pdf) <= 6 and 'DRAFT' in text and 'Reproducibility and references' in text
    for token in ['239.58','223.48','219.74','214.85','207.274224','215.709120','8.434896',
                  '220.98','213.32','209.16','23.84','25.61','10.01','10.29']:
        assert token in text,token
    thumbnails=[]
    for i,page in enumerate(pdf):
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            assert x0>=0 and y0>=0 and x1<=page.rect.width+.1 and y1<=page.rect.height+.1,(i,block)
        target=folder/f'page-{i+1:02d}.png'
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(target)
        with Image.open(target) as image:
            thumb=image.copy();thumb.thumbnail((306,396));thumbnails.append(thumb)
    sheet=Image.new('RGB',(3*326,((len(thumbnails)+2)//3)*426),'#cbd5e1');draw=ImageDraw.Draw(sheet)
    for i,image in enumerate(thumbnails):
        x=(i%3)*326+10;y=(i//3)*426+20;sheet.paste(image,(x,y));draw.text((x,y-14),f'Page {i+1}',fill='black')
    sheet.save(folder/'contact_sheet.png')
    doc=Document(ROOT/f'{NAME}.docx')
    doc_text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
    assert 'DRAFT' in doc_text and 'Reproducibility and references' in doc_text
    assert len(doc.inline_shapes)==1
    for token in ['207.274224','215.709120','209.16','214.85']:
        assert token in doc_text
    # Compare every retained paragraph and table cell across generated formats.
    normalize = lambda value: re.sub(r'\s+', '', value)
    plain = lambda value: re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', value)
    blocks = report_blocks(json.loads((RESULTS/'run_summary.json').read_text()))
    for kind, data in blocks:
        if kind in ('title', 'heading', 'paragraph'):
            expected = [plain(data)]
        elif kind == 'table':
            expected = [str(v) for row in [data[0]] + data[1] for v in row]
        elif kind == 'figure':
            expected = [data[1]]
        else:
            continue
        for value in expected:
            assert normalize(value) in normalize(text), ('PDF', value)
            assert normalize(value) in normalize(doc_text), ('DOCX', value)
    markdown = (ROOT/'REPORT_DRAFT.md').read_text()
    links = re.findall(r'\]\(([^)]+)\)', markdown)
    doc_links = {r.target_ref for r in doc.part.rels.values() if r.reltype.endswith('/hyperlink')}
    for link in links:
        if '://' not in link:
            assert (ROOT/link).is_file(), link
        if not link.endswith('.png'):
            assert link in doc_links, link
    result={'pdf_pages':len(pdf),'pdf_text_inside_page_bounds':True,'key_report_numbers_present':True,
            'docx_readable_and_tables_present':True,'embedded_figures':1,
            'all_paragraphs_and_table_cells_match_source':True,'evidence_links_valid':True,
            'rendered_pages_folder':str(folder.relative_to(ROOT))}
    pdf.close()
    (ASSETS/'document_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def make_archive():
    filenames=['CS664_Flight_Booking.ipynb','run_notebook.py','project_verification.py','requirements.txt',
               'README.md','RESULTS.md','CS664_Project_Wong_Sheung_Wan_Proposal.md']
    files=[ROOT/name for name in filenames]+sorted(p for p in RESULTS.rglob('*') if p.is_file())
    stage=ROOT/'submission_staging'/NAME;stage.mkdir(parents=True,exist_ok=True)
    for source in files:
        relative=source.relative_to(ROOT);target=stage/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert {p.relative_to(stage) for p in stage.rglob('*') if p.is_file()}=={p.relative_to(ROOT) for p in files}
    archive=ROOT/f'{NAME}.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(stage,NAME+'/')
        for file in files:z.write(stage/file.relative_to(ROOT),str(Path(NAME)/file.relative_to(ROOT)))
    with TemporaryDirectory(prefix='cs664-package-check-') as temp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert {Path(name).parts[0] for name in z.namelist()}=={NAME}
            assert all(not any(part in {'.venv','.git','.jupyter','.tools','practice_results'} for part in Path(name).parts)
                       for name in z.namelist())
            assert not any(name.endswith(('.pdf','.docx')) or 'Practice' in name or 'PRACTICE_GUIDE' in name
                           or 'verify_practice.py' in name for name in z.namelist())
            z.extractall(temp)
        extracted=Path(temp)/NAME
        for source in files:assert fingerprint(extracted/source.relative_to(ROOT))==fingerprint(source)
        for filename in ['README.md','RESULTS.md']:
            for link in re.findall(r'\]\(([^)]+)\)',(extracted/filename).read_text()):
                if '://' not in link:assert (extracted/link).exists(),(filename,link)
        nb=nbformat.read(extracted/'CS664_Flight_Booking.ipynb',as_version=4)
        assert len([c for c in nb.cells if c.cell_type=='code' and c.execution_count is not None])==80
        for name in ['run_notebook.py','project_verification.py']:
            compile((extracted/name).read_text(),name,'exec')
        # Exercise the packaged helper without invoking any training.
        import importlib.util
        spec = importlib.util.spec_from_file_location('packaged_verification', extracted/'project_verification.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        fixture = Path(temp)/'run_summary.json'
        fixture.write_text(json.dumps({'mean_cost': 120, 'execution': {'python': '/old/.venv/bin/python'}}))
        original = helper.evidence_digest(fixture)
        fixture.write_text(json.dumps({'mean_cost': 120, 'execution': {'python': '/new/.venv/bin/python'}}))
        assert helper.evidence_digest(fixture) == original
        fixture.write_text(json.dumps({'mean_cost': 121, 'execution': {'python': '/new/.venv/bin/python'}}))
        assert helper.evidence_digest(fixture) != original
        csv_fixture = Path(temp)/'costs.csv'
        csv_fixture.write_text('cost\n120\n')
        original = helper.evidence_digest(csv_fixture)
        csv_fixture.write_text('cost\n180\n')
        assert helper.evidence_digest(csv_fixture) != original
        requirements=(extracted/'requirements.txt').read_text().lower()
        for dependency in ['numpy==','nbformat==','nbclient==','ipykernel==','jupyterlab==','scipy==','matplotlib==']:
            assert dependency in requirements
        result={'status':'PASS','archive_sha256':fingerprint(archive),'files':len(files),
                'top_level_folder':NAME,'crc_check':True,'temporary_extraction':True,
                'byte_identical_copies':True,'saved_80_cell_outputs_present':True,
                'all_markdown_runtime_result_links_present':True,'runtime_modules_compile':True,
                'dependencies_pinned':True,'excluded_materials_absent':True,
                'packaged_fingerprint_relocation_and_numeric_change_fixtures':True,'training_rerun':False}
    (ASSETS/'package_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    import sys
    report_only = '--report-only' in sys.argv
    protected=[ROOT/'CS664_Flight_Booking.ipynb',ROOT/'CS664_Flight_Booking_Practice.ipynb',
               ROOT/'CS664_Project_Wong_Sheung_Wan_Proposal.md', ROOT/'PRESENTATION_SCRIPT.md']+[p for p in RESULTS.rglob('*') if p.is_file()]
    if report_only:
        protected.append(ROOT/f'{NAME}.zip')
    before={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    summary,runtime=audit_saved_evidence()
    make_documents(report_blocks(summary,runtime))
    document_checks=verify_pdf_and_docx()
    package_checks=None if report_only else make_archive()
    assert before=={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    (ASSETS/'preservation_validation.json').write_text(json.dumps({'status':'PASS',
        'protected_files_unchanged':True,'protected_sha256':before,'training_rerun':False},indent=2)+'\n')
    print('Report draft created:',ROOT/f'{NAME}.pdf',ROOT/f'{NAME}.docx')
    print('PDF pages:',document_checks['pdf_pages'],'; rendered for visual inspection.')
    if package_checks:
        print('ZIP verified:',ROOT/f'{NAME}.zip','files:',package_checks['files'])
    print('PASS: every report number audited; main/practice/proposal and all original results unchanged.')


if __name__=='__main__':main()

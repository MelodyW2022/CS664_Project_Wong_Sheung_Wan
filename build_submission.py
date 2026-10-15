"""Create the project report and whitelist-based upload ZIP from saved evidence only.

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
    # Verify the original learned array and actions independently of report formatting.
    q = np.load(RESULTS/'q_values.npy')
    assert q.shape == (7, 5, 2) and np.all(np.isfinite(q))
    assert q.size == 70 and 7 * 5 == 35 and 70 - 5 == 65
    assert np.all(q[0, :, 1] == 0)  # Stored placeholders; never legal actions.
    with np.load(RESULTS/'supplementary_trained_models.npz') as saved:
        match = np.flatnonzero((saved['seeds'] == 66401) & (saved['budgets'] == 50000))
        assert len(match) == 1 and np.array_equal(q, saved['q_tables'][match[0]])
    actions = rows('learned_policy_actions.csv')
    assert len(actions) == 35
    prices = summary['simulation_assumptions']['price_levels']
    for row in actions:
        time, index = int(row['time_remaining']), prices.index(int(row['price']))
        expected = 'BOOK' if time == 0 or q[time, index, 0] >= q[time, index, 1] else 'WAIT'
        assert row['action'] == expected
        if time > 0:
            # Three decimals do not conceal any of the displayed action differences.
            rounded = [float(f'{v:.3f}') for v in q[time, index]]
            assert ('BOOK' if rounded[0] >= rounded[1] else 'WAIT') == expected
    # Interpretation checks: compare the prose findings with complete saved evidence.
    by_state = {(int(r['time_remaining']), int(r['price'])): r['action'] for r in actions}
    assert all(by_state[(time, 120)] == 'BOOK' for time in range(7))
    assert all(by_state[(time, 240)] == ('WAIT' if time >= 5 else 'BOOK') for time in range(7))
    assert by_state[(6, 120)] == 'BOOK' and by_state[(6, 300)] == 'WAIT'
    means = {r['strategy']: float(r['mean_purchase_cost']) for r in rows('mean_cost_comparison.csv')}
    assert min(means, key=means.get) == 'Q-learning (greedy)'
    rules = {k: v for k, v in means.items() if k != 'Q-learning (greedy)'}
    assert min(rules, key=rules.get) == 'Fixed-threshold rule ($180)'
    assert np.isclose(rules['Fixed-threshold rule ($180)'] - means['Q-learning (greedy)'], 4.884)
    budget_means = {int(r['episodes']): float(r['mean_exact_cost']) for r in rows('supplementary_budget_summary.csv')}
    assert f"{budget_means[200000] - budget_means[50000]:.2f}" == '4.11'
    rate_means = {r['condition']: float(r['mean_exact_cost']) for r in rows('learning_dynamics_across_seed_summary.csv')}
    assert min(rate_means, key=rate_means.get) == 'visit_decay'
    checkpoints = rows('learning_dynamics_checkpoints.csv')
    for condition in rate_means:
        for seed in [66401, 66411, 66421, 66431, 66441]:
            recorded = [int(r['completed_episodes']) for r in checkpoints
                        if r['condition'] == condition and int(r['training_seed']) == seed]
            assert recorded == list(range(10000, 200001, 10000))
    logged_output = '\n'.join(o.get('text', '') for cell in code for o in cell.outputs)
    for record in [summary['first_update'], observed]:
        state = tuple(record['state'])
        assert f"episode {record['episode']}, state={state}, action={record['action']}" in logged_output
        assert (f"old={record['old']:.6f}, reward={record['reward']}, "
                f"target={record['target']:.6f}, alpha={record['learning_rate']}") in logged_output
        assert f"= {record['updated']:.6f}" in logged_output
        assert 'next_state' not in record and 'next_q_values' not in record
    illustrative_target = 0 + 1 * (-180)
    illustrative_updated = -200 + .1 * (illustrative_target - (-200))
    assert illustrative_target == -180 and illustrative_updated == -198
    proposal = (ROOT/'CS664_Project_Wong_Sheung_Wan_Proposal.md').read_text()
    thresholds = [(int(p), float(c)) for p, c in re.findall(r'\| \$(\d+) \| \$(\d+\.\d+) \|', proposal)]
    assert len(thresholds) == 5 and min(thresholds, key=lambda item: item[1])[0] == 180
    assert dict(thresholds)[180] == round(float(next(
        row['exact_expected_cost'] for row in rows('exact_mdp_comparison.csv')
        if row['policy'] == 'Fixed-threshold rule ($180)')), 2)
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
                          "BOOK and WAIT records against saved Notebook output", "complete trace", "ten budget runs", "fifteen rate runs",
                          "300 checkpoints/2100 diagnostics", "reported means/ranges", "saved PASS records",
                          "all 70 original Q entries/65 legal pairs and 35 saved greedy actions",
                          "original Q array identical to saved seed-66401/50000 supplementary endpoint",
                          "180 threshold rationale against saved proposal analytical comparison",
                          "Q examples/heatmap patterns, lowest sampled mean and strongest baseline",
                          "budget mean increase and lowest tested learning-rate mean",
                          "all five seeds and twenty checkpoints per learning-rate condition",
                          "prose coverage, rate, action-switch, and update examples"],
             "unavailable_evidence": ["Actual episode-376 update: observed next price", "Actual episode-376 update: historical next-state Q-value snapshot"],
             "report_wait_example": {"type": "illustrative; assumed Q-values, not an actual training log", "reward": 0, "assumed_next_q_book": -180, "assumed_old_current_q_wait": -200, "gamma": 1, "alpha": .1, "target": illustrative_target, "updated": illustrative_updated}}
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
    h("1. Project goal and simulator")
    p("This Python tabular Q-learning application chooses BOOK or WAIT for a simulated traveler with a fixed departure date. It tests whether current fare and time remaining can reduce expected purchase cost relative to simple booking rules. The approach trains a table of BOOK/WAIT values and compares the learned policy (a rule choosing actions from the current state) with three simple booking rules. It also checks the policy against the optimum under the simulator's Markov decision process (MDP), a model of states, actions, and transition probabilities. Supplementary experiments examine training budget and learning rate separately from the original comparison.")
    p("State is (time remaining, current price). Seven daily purchase opportunities run from time 6 to 0; fares are $120, $180, $240, $300, and $360. Each episode starts at one of these five prices, with an equal probability of 20% for each. When the agent waits, the next price is sampled using the probabilities for the current time remaining.")
    table(["Time before WAIT", "+$60", "Unchanged", "−$60"], [["6–3", "30%", "30%", "40%"], ["2–1", "60%", "20%", "20%"]])
    p("WAIT gives reward 0 and reduces time by one. Moves beyond $120–$360 stay at the boundary. BOOK gives the negative current fare as reward and terminates. At time 0, BOOK is mandatory and a requested WAIT becomes BOOK. Actions after purchase raise an error until reset. Refunds, cancellations, and sold-out events are excluded. All probabilities are hypothetical.")
    h("2. Training and Q-value updates")
    p("Q-learning follows Watkins and Dayan (1992). Q(state, action) estimates the total reward from taking that action in that state, then following the best continuation represented by the current Q-table. Each training episode is one simulated trip ending in a purchase. Rewards are negative purchase costs, so a larger Q-value is better. The table starts at zero; the agent learns from observed transitions without receiving their probabilities.")
    table(["Original setting", "Value and meaning"], [
        ["Training episodes / seeds", "50,000 trips; training seed 66401, test seed 66402. Seeds initialize reproducible random-number streams."],
        ["Alpha (learning rate)", "0.1; controls how much each new target changes the current estimate."],
        ["Gamma (discount factor)", "1.0; future rewards are not discounted."],
        ["Epsilon and its schedule", "Epsilon is the probability of a random action during training. It decreases linearly from 1.0 initially to 0.05 by episode 40,000, then stays at 0.05."],
        ["Greedy evaluation", "Choose the largest legal Q-value; ties choose BOOK. Epsilon 0 means no random exploration. No Q updates means the trained table stays fixed throughout evaluation."]], [2.0, 5.2])
    page()
    h("2. Training and Q-value updates (continued)")
    p("Each update changes the CURRENT state/action entry: Q_new = Q_old + alpha × (target − Q_old). The target is the observed reward for terminal BOOK: the trip has ended, so there is no future value. For nonterminal WAIT, target = reward + gamma × the largest legal Q-value at the observed next state. At a deadline next state, only BOOK is legal. The next state is observed after acting; no future price is known in advance.")
    u = summary['first_update']
    h("Actual BOOK update — original training log")
    p(f"Episode {u['episode']}, current state ({u['state'][0]}, ${u['state'][1]}), action BOOK. Observed reward = {u['reward']:g}; old current Q = {u['old']:g}; target = {u['target']:g}; alpha = {u['learning_rate']:g}; updated current Q = {u['updated']:g}. This purchase ends the episode, so no next-state value is used. Arithmetic: 0 + 0.1 × (−120 − 0) = −12.")
    p("The BOOK record is saved in results/run_summary.json. No complete actual WAIT update with its observed next state and historical next-state Q-values was found in the saved logs, Notebook outputs, or result artifacts. The example below uses assumptions, not reconstructed training data.")
    h("Illustrative WAIT update — assumed Q-values, not an actual training log")
    table(["Component", "Hypothetical value"], [
        ["Current state / action", "(1, $240) / WAIT"],
        ["Observed next state in this example", "(0, $180)"],
        ["Immediate reward", "0"],
        ["Assumed next-state legal Q-values", "Q(BOOK) = −180; WAIT is illegal at the deadline."],
        ["Assumed old current Q(WAIT)", "−200"],
        ["Gamma / alpha", "1 / 0.1"],
        ["Target", "0 + 1 × (−180) = −180"],
        ["Updated current Q(WAIT)", "−200 + 0.1 × [−180 − (−200)] = −198"]], [2.7, 4.5])
    p("This update changes Q((1, $240), WAIT), not the next-state entry. Ticket prices are limited to $120–$360, while Q-values are learned reward estimates and need not equal a valid ticket price. The old value −200 and next-state value −180 here are assumed estimates.")
    p("The complete table on the next page contains final values after 50,000 training episodes, from results/q_values.npy. There are 7 time values × 5 prices × 2 actions = 70 stored entries, with 65 legal state/action pairs. Deadline WAIT is not allowed. Values are rounded to three decimals; greedy actions use unrounded values. This final learned table is separate from the logged BOOK update and illustrative WAIT example above. Even BOOK estimates can differ from the known negative fare after finite training.")
    q_examples = np.load(RESULTS/'q_values.npy')
    p(f"For example, at (6, $120), Q(BOOK) = {q_examples[6, 0, 0]:.3f} exceeds Q(WAIT) = {q_examples[6, 0, 1]:.3f}, so the policy BOOKs. At (6, $300), Q(WAIT) = {q_examples[6, 3, 1]:.3f} exceeds Q(BOOK) = {q_examples[6, 3, 0]:.3f}, so it WAITs. The larger (less negative) legal estimate determines the action; these estimates do not establish optimality.")
    page()
    h("2. Complete original trained Q-table")
    q = np.load(RESULTS/'q_values.npy')
    prices = summary['simulation_assumptions']['price_levels']
    table(["Time remaining", "Price", "Q(BOOK)", "Q(WAIT)", "Greedy action"], [
        [str(time), money(price, 0), f"{q[time, index, 0]:.3f}",
         "Not allowed" if time == 0 else f"{q[time, index, 1]:.3f}",
         "BOOK" if time == 0 or q[time, index, 0] >= q[time, index, 1] else "WAIT"]
        for time in range(6, -1, -1) for index, price in enumerate(prices)], [1.4, 1.0, 1.6, 1.6, 1.6])
    page()
    h("3. Original sampled four-strategy comparison")
    p("Only Q-learning was trained: 50,000 episodes with seed 66401. The three baselines are fixed rules requiring no training. All four used the same 10,000 fresh seven-price test paths, generated with NumPy seed 66402. Only the evaluator sees full paths; policies see the current state and book by the deadline. The $180 threshold was the cheapest of the five fare-level fixed thresholds under the known model, without using test results. The learning agent does not receive that model.")
    table(["Strategy", "Rule", "Sample mean cost"], [[r['strategy'], rule, money(r['mean_purchase_cost'])] for r, rule in zip(rows('mean_cost_comparison.csv'), ["BOOK first day", "BOOK at $120; else deadline", "BOOK at $180 or below; else deadline", "Original greedy policy"])], [2.3, 3.4, 1.5])
    p("Q-learning had the lowest sampled mean ($214.85), with mean paired savings of $4.88 versus the strongest rule, the $180 threshold.")
    table(["Baseline", "Mean paired saving", "Individual paired 95% CI"], [[r['baseline'], money(r['mean_paired_saving']), f"[{money(r['ci_95_lower'])}, {money(r['ci_95_upper'])}]" ] for r in rows('paired_savings_95ci.csv')], [2.7, 1.7, 2.8])
    p("For 10,000 test paths, paired saving is baseline cost minus Q-learning cost on the same path. Each 95% confidence interval (CI) places a margin around the mean: the paired sample standard deviation divided by the square root of 10,000, multiplied by SciPy's Student-t multiplier (97.5th percentile; 9,999 degrees of freedom). All endpoints are positive, supporting mean savings conditional on this trained policy. These large-sample intervals do not describe individual-trip savings, training variation, or simultaneous coverage across baselines.")
    p("Evidence: results/mean_cost_comparison.csv, results/paired_savings_95ci.csv, and results/paired_purchase_costs.csv.")
    h("4. Exact model expectations")
    p("Backward induction starts at the deadline fare and works backward, choosing booking or the probability-weighted cost of waiting with optimal continuation. Fixed-policy evaluation follows a policy's actions instead. Both average over the five equally likely starting fares. These are analytical expectations to floating-point precision, not sampled means or Q-estimates. The optimality gap is learned-policy expected cost minus optimal expected cost.")
    table(["Analytical reference", "Expected cost"], [["Optimal finite-horizon policy", money(exact['optimal_expected_cost'], 6)], ["Original learned policy", money(exact['learned_policy_expected_cost'], 6)], ["Learned minus optimal", money(exact['learned_policy_expected_cost'] - exact['optimal_expected_cost'], 6)]], [4.5, 2.7])
    p("The learned policy remains $8.43 above optimal per trip: at (6, $180), it BOOKs while the optimum WAITs. Its $214.85 sample mean estimates the $215.709120 model cost with sampling variation. Evidence: results/exact_mdp_comparison.csv and results/exact_state_values.csv.")
    page()
    h("5. Original learned policy and complete decision trace")
    blocks.append(("figure", (str(RESULTS / 'policy_heatmap.png'), "Figure 1. Original greedy policy: BOOK or WAIT at each state. Time 0 is mandatory BOOK.", 6.4, 4.8)))
    p("The heatmap shows time-dependent decisions: at $240, WAIT at times 6–5 changes to BOOK at 4–0; $120 is always BOOK. This describes the learned rule, not proof of optimality. The complete trace is path index 1, the first test example with at least three decisions; unseen post-purchase prices are excluded.")
    table(["Day", "Time remaining", "Price", "Action", "Reward"], [[r['day'], r['time_remaining'], money(r['price'], 0), r['action'], r['reward']] for r in rows('q_learning_trace.csv')])
    p(f"The episode ends after {len(rows('q_learning_trace.csv'))} decisions with a {money(summary['final_purchase_cost'], 0)} purchase. It illustrates policy behavior, not average performance: positive mean savings can coexist with expensive trips. Evidence: results/q_learning_trace.csv and results/learned_policy_actions.csv.")
    page()
    h("6. Executed verification")
    table(["Core check", "Cases", "Outcome"], [[r['check'], f"{int(r['cases']):,}", r['status']] for r in rows('verification_results.csv')], [4.8, 1.0, 1.4])
    p("The exact reference was independently checked by enumerating all 3,645 possible starting-price/move sequences for each of five policies (18,225 comparisons), weighted by their model probabilities. Supplementary checks passed for nonzero-old-value Q updates, deadline selection, all-state rewards and invalid actions, correct identification of optimal-action ties, independent training, and preservation of the original model. Actions with equally low exact expected costs are treated as optimal ties.")
    p("Additional logging checks verified update counts, applied learning rates, terminal/legal targets, and evaluation at checkpoints. At all 300 checkpoints (saved training snapshots), evaluation left Q, update counts, and the random-number generator state unchanged. Constant alpha 0.1 exactly reproduced the ten earlier 50,000/200,000 endpoint Q-tables, visit counts, and analytical costs. The saved main Notebook has 80 successfully executed code cells. These finite checks support implementation behavior; they do not prove convergence.")
    p("Complete executed records: results/verification_results.csv, results/supplementary_verification.csv, and results/learning_dynamics_verification.csv.")
    h("7. Supplementary training-budget experiment")
    p("Seeds 66401, 66411, 66421, 66431, and 66441 each restarted from zero at 50,000 and 200,000 episodes (ten runs). Only episode budget changed: simulator, alpha 0.1, gamma 1, initialization, and the original epsilon schedule stayed fixed. Final greedy policies were evaluated exactly under the uniform initial-fare distribution.")
    table(["Episodes", "Mean exact cost", "Across-seed cost range"], [[f"{int(r['episodes']):,}", money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]" ] for r in rows('supplementary_budget_summary.csv')], [1.5, 2.2, 3.5])
    pairs = rows('supplementary_paired_gap_changes.csv')
    example = next(r for r in pairs if r['training_seed'] == '66401')
    increase = float(example['gap_200000']) - float(example['gap_50000'])
    budget_summary = {int(r['episodes']): float(r['mean_exact_cost']) for r in rows('supplementary_budget_summary.csv')}
    mean_increase = budget_summary[200000] - budget_summary[50000]
    p(f"All five seeds worsened: mean exact cost rose from {money(budget_summary[50000])} to {money(budget_summary[200000])} (+{money(mean_increase)}). More episodes alone did not improve these constant-0.1 runs; seed 66401 increased by {money(increase)}. All ten models cost less than the immediate and $120 baselines in exact evaluation; seven cost less than the $180 baseline. All 65 legal pairs were visited, yet seven original-policy states were suboptimal. None had an action in the bottom quarter of visit counts; this does not explain the gap.")
    p("Full evidence: results/supplementary_training_runs.csv, results/supplementary_paired_gap_changes.csv, results/supplementary_suboptimal_states.csv, and CS664_Flight_Booking.ipynb.")
    page()
    h("8. Supplementary learning-rate experiment")
    p("The five seeds restarted independently for 200,000 episodes under constant alpha 0.1, constant alpha 0.03, or alpha(s,a) = 0.1/(1 + N_previous(s,a)/1000). N_previous counts prior updates to that pair. With 0, 1,000, and 3,000 prior updates, alpha is 0.1, 0.05, and 0.025. Scale 1,000 was prespecified. Other settings and the legal greedy evaluator were unchanged.")
    p("Fifteen runs recorded 300 training checkpoints and 2,100 state records for the seven originally suboptimal states. Equal seeds do not ensure identical experience across conditions: changed policies can alter later experiences and consumption of random numbers. Every run, including equal-cost policies, remains in the evidence.")
    labels = {'constant_0.1': 'Constant 0.1', 'constant_0.03': 'Constant 0.03', 'visit_decay': 'Visit decay'}
    rate_summary = rows('learning_dynamics_across_seed_summary.csv')
    table(["Condition", "Mean exact cost", "Across-seed cost range", "Mean optimality gap"], [[labels[r['condition']], money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]", money(r['mean_gap'])] for r in rate_summary], [1.5, 1.5, 2.7, 1.5])
    comparison = next(r for r in rows('learning_dynamics_paired_comparisons.csv') if r['training_seed'] == '66401')
    reduction03 = float(comparison['cost_alpha_0.1']) - float(comparison['cost_alpha_0.03'])
    reduction_decay = float(comparison['cost_alpha_0.1']) - float(comparison['cost_visit_decay'])
    rate_means = {r['condition']: float(r['mean_exact_cost']) for r in rate_summary}
    decay_gap = next(float(r['mean_gap']) for r in rate_summary if r['condition'] == 'visit_decay')
    p(f"Visit decay had the lowest tested mean exact cost ({money(rate_means['visit_decay'])}, versus {money(rate_means['constant_0.1'])} for constant 0.1), but its mean optimality gap remained {money(decay_gap)}. For each of the five seeds, both alternative learning rates lowered final exact cost relative to constant alpha 0.1. For seed 66401, reductions were {money(reduction03)} with alpha 0.03 and {money(reduction_decay)} with visit decay. Every final policy still had a positive optimality gap. Checkpoint costs fluctuated in all conditions: for example, seed 66401 under alpha 0.1 worsened by $4.72 from 20,000 to 30,000 episodes. Action switches accompanied performance changes, but sparse checkpoints do not isolate their cause. No best-checkpoint selection was performed.")
    p("Full run and checkpoint evidence: results/learning_dynamics_final_comparison.csv, results/learning_dynamics_paired_comparisons.csv, and results/learning_dynamics_checkpoints.csv. The saved checkpoint plot, results/learning_dynamics_checkpoint_gaps.png, shows all five seeds per condition; lines join recorded evaluations only. RESULTS.md indexes state diagnostics.")
    h("9. Limitations and validation answers")
    p("Cost reduction: yes in the original simulator test sample, relative to all three baselines; an individual trip need not save money. Reliability: the paired intervals support positive mean savings for this fixed policy; the exploratory seed studies do not establish universal robustness. Optimality and correctness: the original policy remains above the exact optimum, while saved verification supports the stated implementation without proving convergence.")
    p("The hypothetical probabilities, five fares, short horizon, and omission of sold-out events provide no evidence of real-flight savings. Across-seed ranges are descriptive, not confidence intervals; two budgets do not establish a continuous trend. The supplementary analyses are exploratory, not independent confirmation. Learning rates can affect target variability, learning speed, and state/action coverage, but their causal contributions were not isolated. No additional tuning was performed.")
    p("Project evidence files are included in the submitted ZIP; paths are relative to its top-level folder. README.md provides setup and execution instructions; RESULTS.md indexes complete saved outputs. Monetary summaries are rounded after full-precision calculations; saved artifacts retain their precision.")
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
        # Only public reference URLs are active links; evidence paths are plain text.
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
            padding = 3 if headers == ["Time remaining", "Price", "Q(BOOK)", "Q(WAIT)", "Greedy action"] else 5
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#184e77")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f1f5f9")]),
                ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),
                ("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),padding),
                ("BOTTOMPADDING",(0,0),(-1,-1),padding),("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#184e77"))]))
            flow.extend([table,Spacer(1,8)])
        elif kind=="figure":
            path,caption,width,height=data
            flow.extend([Image(path,width=width*inch,height=height*inch),Paragraph(escape(caption),styles["CaptionReport"])])
        elif kind=="pagebreak":flow.append(PageBreak())
    def footer(canvas,doc):
        canvas.setFont("Report",8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(7.85*inch,.35*inch,f"Page {doc.page}")
    pdf=ROOT/f"{NAME}.pdf"
    SimpleDocTemplate(str(pdf),pagesize=(8.5*inch,11*inch),leftMargin=.65*inch,rightMargin=.65*inch,
        topMargin=.6*inch,bottomMargin=.65*inch,title="Flight Booking Decision Assistant",author="Prepared for Sheung Wan Wong").build(flow,onFirstPage=footer,onLaterPages=footer)

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
    document.core_properties.title="Flight Booking Decision Assistant"
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
    paragraph.text='Page '
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


def audit_exported_hyperlinks(pdf, doc):
    """Inspect actual PDF action dictionaries and every DOCX relationship part."""
    from lxml import etree
    public = set()  # The report intentionally contains no active hyperlinks.
    pdf_targets = []
    for number, page in enumerate(pdf, 1):
        for link in page.get_links():
            xref = link['xref']
            assert pdf.xref_get_key(xref, 'A/S') == ('name', '/URI'), (number, link)
            value_type, target = pdf.xref_get_key(xref, 'A/URI')
            assert value_type == 'string' and target in public, (number, target)
            pdf_targets.append({'page': number, 'target': target,
                                'raw_annotation': pdf.xref_object(xref)})
    assert {row['target'] for row in pdf_targets} == public
    docx_targets = []
    with zipfile.ZipFile(ROOT/f'{NAME}.docx') as archive:
        for part in archive.namelist():
            if not part.endswith('.rels'):
                continue
            for relation in etree.fromstring(archive.read(part)):
                if relation.get('TargetMode') == 'External':
                    target = relation.get('Target')
                    assert relation.get('Type').endswith('/hyperlink') and target in public, (part, target)
                    docx_targets.append({'part': part, 'target': target})
    assert {row['target'] for row in docx_targets} == public
    return {'status': 'PASS', 'pdf_action_dictionaries_inspected': True,
            'docx_all_relationship_parts_inspected': True,
            'pdf_links': pdf_targets, 'docx_external_relationships': docx_targets,
            'no_active_hyperlinks': True, 'training_rerun': False}


def verify_pdf_and_docx():
    import pymupdf as fitz
    from docx import Document
    from PIL import Image,ImageOps,ImageDraw
    folder=ROOT/'.cache'/'report_preview';folder.mkdir(parents=True,exist_ok=True)
    pdf=fitz.open(ROOT/f'{NAME}.pdf')
    text='\n'.join(page.get_text() for page in pdf)
    assert len(pdf) > 0 and not re.search(r'(?i)\bdraft\b|awaiting student review', text) and '9. Limitations' in text
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
    assert not re.search(r'(?i)\bdraft\b|awaiting student review', doc_text) and '9. Limitations' in doc_text
    assert len(doc.inline_shapes)==1
    for token in ['207.274224','215.709120','209.16','214.85']:
        assert token in doc_text
    # Check all Q-table rows in their exported order, not just individual tokens.
    q_headers = ['Time remaining', 'Price', 'Q(BOOK)', 'Q(WAIT)', 'Greedy action']
    q = np.load(RESULTS/'q_values.npy')
    prices = json.loads((RESULTS/'run_summary.json').read_text())['simulation_assumptions']['price_levels']
    expected_q_rows = [[str(time), money(price, 0), f'{q[time, index, 0]:.3f}',
                        'Not allowed' if time == 0 else f'{q[time, index, 1]:.3f}',
                        'BOOK' if time == 0 or q[time, index, 0] >= q[time, index, 1] else 'WAIT']
                       for time in range(6, -1, -1) for index, price in enumerate(prices)]
    q_doc_tables = [table for table in doc.tables if [c.text for c in table.rows[0].cells] == q_headers]
    assert len(q_doc_tables) == 1
    assert [[c.text for c in row.cells] for row in q_doc_tables[0].rows[1:]] == expected_q_rows
    from docx.oxml.ns import qn
    assert q_doc_tables[0].rows[0]._tr.get_or_add_trPr().find(qn('w:tblHeader')) is not None
    q_pdf_values = []
    q_table_pages = []
    for number, page in enumerate(pdf, 1):
        lines = page.get_text().splitlines()
        if all(header in lines for header in q_headers):
            q_table_pages.append(number)
            start = lines.index('Greedy action') + 1
            q_pdf_values.extend(lines[start:])
    assert q_pdf_values == [cell for row in expected_q_rows for cell in row]
    assert all(bad not in text and bad not in doc_text for bad in ['Å~', 'Ã—', '\ufffd'])
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
    hyperlink_checks = audit_exported_hyperlinks(pdf, doc)
    markdown = (ROOT/'REPORT_DRAFT.md').read_text()
    # Validate every displayed project path against the existing submitted archive.
    evidence_paths = sorted(set(re.findall(
        r'(?<![\w/])[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:csv|json|ipynb|md|png|npy|npz)\b', markdown)))
    with zipfile.ZipFile(ROOT/f'{NAME}.zip') as archive:
        for path in evidence_paths:
            assert f'{NAME}/{path}' in archive.namelist(), path
    hyperlink_checks['zip_contained_plain_text_paths'] = evidence_paths
    (ASSETS/'hyperlink_validation.json').write_text(json.dumps(hyperlink_checks, indent=2) + '\n')
    result={'pdf_pages':len(pdf),'pdf_text_inside_page_bounds':True,'key_report_numbers_present':True,
            'docx_readable_and_tables_present':True,'embedded_figures':1,
            'all_paragraphs_and_table_cells_match_source':True,'evidence_links_valid':True,
            'all_35_exported_q_rows_checked_against_saved_array':True,
            'q_table_pdf_pages':q_table_pages,'docx_q_table_repeating_header':True,
            'visible_encoding_errors_absent':True,
            'rendered_pages_folder':str(folder.relative_to(ROOT))}
    pdf.close()
    (ASSETS/'document_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def make_archive():
    """Package runtime and complete saved experiment evidence, without training."""
    filenames = ['CS664_Flight_Booking.ipynb', 'run_notebook.py', 'project_verification.py',
                 'requirements.txt', 'README.md', 'RESULTS.md']
    files = [ROOT/name for name in filenames] + sorted(p for p in RESULTS.rglob('*') if p.is_file())
    archive = ROOT/f'{NAME}.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            z.write(file, str(Path(NAME)/file.relative_to(ROOT)))
    with TemporaryDirectory(prefix='cs664-package-check-') as temp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert not any(n.endswith(('.pdf', '.docx')) for n in z.namelist())
            assert set(z.namelist()) == {str(Path(NAME)/p.relative_to(ROOT)) for p in files}
            z.extractall(temp)
        extracted = Path(temp)/NAME
        for source in files:
            assert fingerprint(extracted/source.relative_to(ROOT)) == fingerprint(source)
        for filename in ['README.md', 'RESULTS.md']:
            for link in re.findall(r'\]\(([^)]+)\)', (extracted/filename).read_text()):
                if '://' not in link:
                    assert (extracted/link).exists(), (filename, link)
        nb = nbformat.read(extracted/'CS664_Flight_Booking.ipynb', as_version=4)
        nbformat.validate(nb)
        code = [c for c in nb.cells if c.cell_type == 'code']
        assert len(code) == 80
        assert [c.execution_count for c in code] == list(range(1, 81))
        assert not any(o.output_type == 'error' for c in code for o in c.outputs)
        import ast
        imports = set()
        for filename in ['run_notebook.py', 'project_verification.py']:
            source = (extracted/filename).read_text()
            compile(source, filename, 'exec')
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module)
        for i, cell in enumerate(code):
            compile(cell.source, f'Notebook code cell {i+1}', 'exec')
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add(node.module)
        # Exercise only the runner's optional-file preflight, never client.execute().
        tree = ast.parse((extracted/'run_notebook.py').read_text())
        start = next(i for i,n in enumerate(tree.body)
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'proposal' for t in n.targets))
        end = next(i for i,n in enumerate(tree.body)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'before' for t in n.targets))
        preflight = ast.Module(body=tree.body[start:end+1], type_ignores=[])
        context = {'ROOT': extracted, 'hashlib': hashlib}
        exec(compile(preflight, 'packaged optional-file preflight', 'exec'), context)
        assert context['originals'] == [] and context['before'] == {}
        import subprocess, sys
        # Existing interpreter only: no clean installation or full execution is claimed.
        check_script = """
import importlib, importlib.metadata, json
from pathlib import Path
from packaging.requirements import Requirement
modules = json.loads(Path('import_inventory.json').read_text())
for module in modules:
    importlib.import_module(module)
for line in Path('requirements.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    requirement = Requirement(line)
    if requirement.marker is None or requirement.marker.evaluate():
        version = importlib.metadata.version(requirement.name)
        assert version in requirement.specifier, (requirement.name, version, str(requirement.specifier))
print('PASS: all discovered imports and installed pinned requirement versions')
"""
        (extracted/'import_inventory.json').write_text(json.dumps(sorted(imports)))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR=str(Path(temp)/'matplotlib'), XDG_CACHE_HOME=str(Path(temp)/'cache'))
        check = subprocess.run([sys.executable, '-c', check_script], cwd=extracted, env=env,
                               capture_output=True, text=True, check=True)
        dependency_check = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=extracted,
                                         env=env, capture_output=True, text=True, check=True)
        result = {'status': 'PASS', 'archive_sha256': fingerprint(archive), 'files': len(files),
                  'manifest': [str(p.relative_to(ROOT)) for p in files],
                  'top_level_folder': NAME, 'crc_check': True, 'temporary_extraction': True,
                  'byte_identical_copies': True, 'report_pdf_and_docx_excluded_for_separate_submission': True, 'saved_80_cell_outputs_present': True,
                  'all_markdown_runtime_result_links_present': True, 'notebook_and_modules_compile': True,
                  'discovered_imports': sorted(imports), 'imports_succeeded_in_extracted_folder': True,
                  'installed_versions_match_pinned_requirements': True,
                  'pip_check': dependency_check.stdout.strip(),
                  'optional_proposal_and_instructor_absence_preflight': True,
                  'excluded_materials_absent': True, 'training_rerun': False,
                  'fresh_full_execution': False, 'clean_dependency_installation_performed': False}
    (ASSETS/'package_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    import sys
    report_only = '--report-only' in sys.argv
    protected=[ROOT/'CS664_Flight_Booking.ipynb',ROOT/'CS664_Flight_Booking_Practice.ipynb',
               ROOT/'CS664_Project_Wong_Sheung_Wan_Proposal.md', ROOT/'PRESENTATION_SCRIPT.md',
               ROOT/'RESULTS.md', ROOT/'run_notebook.py', ROOT/'project_verification.py',
               ROOT/'requirements.txt']+[p for p in RESULTS.rglob('*') if p.is_file()]
    if report_only:
        protected.append(ROOT/f'{NAME}.zip')
    before={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    summary,runtime=audit_saved_evidence()
    make_documents(report_blocks(summary,runtime))
    package_checks=None if report_only else make_archive()
    document_checks=verify_pdf_and_docx()
    assert before=={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    (ASSETS/'preservation_validation.json').write_text(json.dumps({'status':'PASS',
        'protected_files_unchanged':True,'protected_sha256':before,'training_rerun':False},indent=2)+'\n')
    print('Report created:',ROOT/f'{NAME}.pdf',ROOT/f'{NAME}.docx')
    print('PDF pages:',document_checks['pdf_pages'],'; rendered for visual inspection.')
    if package_checks:
        print('ZIP verified:',ROOT/f'{NAME}.zip','files:',package_checks['files'])
    print('PASS: every report number audited; main/practice/proposal and all original results unchanged.')


if __name__=='__main__':main()

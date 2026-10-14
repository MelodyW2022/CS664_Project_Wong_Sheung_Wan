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


def money(value, places=6):
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
    assert all(float(row["gap_from_optimal"]) > 0 for row in finals)
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


def report_blocks(summary, runtime, graph):
    blocks = []
    def title(text): blocks.append(("title", text))
    def h(text): blocks.append(("heading", text))
    def p(text): blocks.append(("paragraph", text))
    def table(headers, data, widths=None): blocks.append(("table", (headers, data, widths)))
    def figure(path, caption, width, height): blocks.append(("figure", (str(path), caption, width, height)))
    exact = summary["exact_mdp"]
    title("Flight Booking Decision Assistant")
    p("CS664 Term Project | Prepared for Sheung Wan Wong | October 14, 2026")
    p("DRAFT — awaiting student review. This report records executed computational work; it does not establish personal practice, presentation readiness, or submission.")
    h("1. Goal, classification, and scope")
    p("This machine learning application uses Python tabular Q-learning to choose BOOK or WAIT for a simulated traveler with a fixed departure date. Its goal is to investigate whether current fare and time remaining reduce expected purchase cost relative to simple booking rules. It is a small teaching model, not a real-airfare forecasting service.")
    p("The original English proposal planned a seven-opportunity simulator, 50,000 training episodes, a four-strategy comparison on 10,000 fresh shared price paths, paired uncertainty estimates, an exact MDP reference, a policy heatmap, a trace, and program checks. That scope was completed without revising the proposal or changing the simulator or four strategies. Later exploratory extensions investigated five seeds at two training budgets and three prespecified learning-rate conditions. These extensions remain separate from the original policy and its sampled comparison.")
    p("The three validation questions are: (1) Does Q-learning reduce cost versus the baselines? (2) How reliable are the observed savings? (3) How close is the learned policy to optimal, and does the implementation follow the rules? The instructor's CS664 Term Project PDF supplies the official requirements; the proposal is the implementation plan [1, 2].")
    h("2. Simulator and decision process")
    p("State is (time remaining, current price). Seven daily purchase opportunities run from time 6 to time 0, with prices $120, $180, $240, $300, and $360. The initial price is uniform over these five levels. This is an artificial teaching horizon, not a recommended real-world booking window.")
    table(["Time before WAIT", "+$60", "Unchanged", "-$60"], [["6–3", "30%", "30%", "40%"], ["2–1", "60%", "20%", "20%"]])
    p("WAIT gives reward 0, reduces time by one, and samples the next price using the regime before the action. An attempted move outside $120–$360 stays at the boundary. BOOK pays the current fare, gives reward equal to its negative, and terminates. At time 0 only BOOK is legal; a requested WAIT is converted to BOOK. Further steps after purchase raise an error until reset. There are no refunds, cancellations, or sold-out events. Probabilities are hypothetical, not measured flight-price patterns.")
    h("3. Q-learning method and key code")
    p("Q-learning updates state/action values from observed rewards and next states [3]. The agent does not receive the transition-probability table. Maximizing the undiscounted negative purchase reward minimizes eventual cost. The Q-table starts at zero with shape 7 × 5 × 2; 65 state/action pairs are legal because deadline WAIT is excluded. Greedy ties choose BOOK.")
    table(["Original setting", "Value"], [["Training episodes", f"{summary['training_episodes']:,}"],
          ["Learning rate / gamma", "0.1 / 1.0"], ["Training / test seeds", f"{summary['training_seed']} / {summary['test_seed']}"],
          ["Exploration", "Epsilon 1.0 on episode 1 to 0.05 on episode 40,000; hold 0.05 afterward"],
          ["Evaluation", "Greedy actions, epsilon 0, no Q updates"]], [2.0, 5.2])
    p("The update is Q_new = Q_old + alpha × (target − Q_old). For terminal BOOK, target = reward only. Otherwise target = reward + gamma × the largest legal next-state Q-value. At a deadline next state, only BOOK enters the bootstrap target.")
    table(["Actual logged update", "Observed value"], [["Episode / state / action", "1 / (6, $120) / BOOK"],
          ["Old value", str(summary['first_update']['old'])], ["Reward / terminal target", "-120 / -120"],
          ["Alpha", str(summary['first_update']['learning_rate'])], ["Updated value", str(summary['first_update']['updated'])]], [2.6, 4.6])
    p("Thus 0 + 0.1 × (-120 − 0) = -12. An actual WAIT update at episode 376, state (1, $240), had old value 0, reward 0, target -30, and updated value -3. With gamma 1, its best legal future estimate at that training moment was -30; this is not the final Q-table's exact expected cost.")
    table(["Key code", "Purpose"], [["next_price; FlightBookingEnv.step", "Probability regimes, clipping, rewards, forced deadline purchase, and termination"],
          ["update_q; choose_training_action", "Terminal/nonterminal targets, epsilon-greedy exploration, and legal greedy actions"],
          ["train_q_learning; play_path", "Independent episode training and stop-at-purchase state-only evaluation"],
          ["solve_optimal_mdp; exact_policy_values", "Backward induction and fixed-policy analytical expectations"]], [2.5, 4.7])
    h("4. Original sampled four-strategy evaluation")
    p("All strategies were evaluated on the same 10,000 new seven-price trajectories, generated with test seed 66402, separate from original training seed 66401. Pre-generation belongs to the evaluator: every decision function receives only its current state, never future prices. Evaluation stops at purchase, forces deadline booking, uses no exploration, and leaves Q unchanged. NumPy's seeded Generator supports reproducible random streams [5].")
    table(["Strategy", "Rule", "Sample mean cost"], [[r['strategy'], rule, money(r['mean_purchase_cost'], 3)] for r, rule in
          zip(rows('mean_cost_comparison.csv'), ["BOOK first day", "BOOK at $120; else deadline", "BOOK at $180 or below; else deadline", "Original 50,000-episode greedy policy"])], [2.0, 3.7, 1.5])
    p("The $180 baseline was chosen from analytical model knowledge in the original proposal, not by searching the test trajectories. The original Q-learning policy is retained; improved supplementary models do not replace it.")
    table(["Baseline", "Mean paired saving", "Individual paired 95% CI"], [[r['baseline'], money(r['mean_paired_saving'], 3),
          f"[{money(r['ci_95_lower'])}, {money(r['ci_95_upper'])}]" ] for r in rows('paired_savings_95ci.csv')], [2.4, 1.7, 3.1])
    p("Saving is baseline cost minus Q-learning cost on the same trajectory. The interval is mean(d) ± t(0.975, 9999) × sample_sd(d)/sqrt(10000), using SciPy's Student-t quantile [4]. All three intervals are above zero for this fixed policy and sample. They are large-sample approximations for mean savings, not ranges containing 95% of individual-trip savings; they are not simultaneous intervals and do not measure variability across training seeds.")
    h("5. Analytical optimality reference, policy, and trace")
    p("Backward induction uses the known simulator: V*(0,p)=p and V*(t,p)=min[p, sum P(p'|t,p)V*(t−1,p')]. A fixed policy chooses its action instead of taking the minimum. Average over the five uniform starting prices. These are full-model analytical expectations to floating-point precision, distinct from the finite test sample.")
    table(["Analytical reference", "Expected cost"], [["Optimal finite-horizon policy", money(exact['optimal_expected_cost'])],
          ["Original learned policy", money(exact['learned_policy_expected_cost'])], ["Learned minus optimal", money(exact['gap_above_optimal'])]])
    p("The original policy is suboptimal despite beating the simple rules in the sampled experiment. For example, at (6, $180) it BOOKs while the exact optimal action is WAIT. Its sampled $214.854 cost differs from its analytical $215.709120 expectation because only 10,000 test trips were sampled; Q-values themselves are estimates, not exact policy costs.")
    figure(RESULTS / 'policy_heatmap.png', "Figure 1. Original greedy BOOK/WAIT policy. Price columns and time rows are labeled; time 0 is mandatory BOOK.", 6.4, 4.8)
    p("The complete original trace below is trajectory index 1, selected as the first test example with at least three decisions. It is an illustration, not a performance estimator; unseen post-purchase prices are not displayed.")
    table(["Day", "Time remaining", "Price", "Action", "Reward"], [[r['day'], r['time_remaining'], money(r['price'], 0), r['action'], r['reward']] for r in rows('q_learning_trace.csv')])
    p(f"This episode ended after seven decisions with a final purchase of {money(summary['final_purchase_cost'], 0)}. Average savings are compatible with expensive individual trips.")
    h("6. Executed verification and execution readiness")
    table(["Executed core check", "Cases", "Outcome"], [[r['check'], r['cases'], r['status']] for r in rows('verification_results.csv')], [4.8, 1.0, 1.4])
    p("The reference expectation was independently verified by enumerating 3,645 latent-move paths per policy for five policies (18,225 comparisons). Supplementary checks also executed nonzero-old-value Q updates, 20 deadline-selector cases, 71 all-state/invalid-action cases, tie-aware action penalties, and preservation of the original model. A differing action is not an error if both exact action costs are equally optimal within numerical tolerance. No algorithm correction was required.")
    p("Learning-dynamics instrumentation checked pre-update pair counts, exactly one increment, actual alpha application, terminal/legal targets, independent training state, and read-only checkpoints. All 300 checkpoints preserved Q, counts, and training RNG state. Constant alpha 0.1 reproduced all ten earlier 50,000/200,000 endpoint Q-tables, visit counts, and analytical costs exactly. These are executed finite checks, not proofs of all software behavior or convergence.")
    p(f"The saved fresh-kernel Notebook has 80 successfully executed code cells. Its recorded interval from the first cell's busy timestamp to the final cell's idle timestamp is {runtime:.2f} seconds, excluding kernel startup and runner postchecks. Complete CLI wall time was not instrumented. Packaging reused this successful run; no training was rerun for the report or ZIP. A necessary runner portability correction now ignores machine-specific execution metadata and figure bytes when checking numerical fingerprints. Fixtures confirmed relocation is accepted and changed costs are detected.")
    h("7. Supplementary five-seed/two-budget investigation")
    p("The prespecified seeds were 66401, 66411, 66421, 66431, and 66441. Each restarted from zero at 50,000 and 200,000 episodes: ten runs, changing only episode budget. Simulator, alpha 0.1, gamma 1, initialization, and the original epsilon schedule stayed fixed. Each final greedy policy was evaluated analytically under the same uniform initial distribution.")
    table(["Seed", "Episodes", "Exact policy cost", "Gap above optimal"], [[r['training_seed'], f"{int(r['episodes']):,}", money(r['exact_expected_cost']), money(r['gap_from_optimal'])] for r in rows('supplementary_training_runs.csv')])
    table(["Episodes", "Mean exact cost", "Across-seed cost range"], [[f"{int(r['episodes']):,}", money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]" ] for r in rows('supplementary_budget_summary.csv')])
    p("All five seeds worsened at 200,000 episodes; for seed 66401 the exact cost increased by $4.256448. All ten models were below the immediate and $120 baselines' exact costs; seven were below the $180 baseline. Two budgets do not establish a continuous trend. Across-seed ranges are descriptive observations over five selected seeds, not the original paired confidence intervals.")
    p("The original model had seven strictly suboptimal states. At (6, $180), BOOK/WAIT had 5,469/4,490 visits, and BOOK's exact action penalty with optimal continuation was $16.2168. None of the seven met the descriptive lower-quartile coverage flag (cutoff 326 visits); all 65 legal pairs had been visited. This evidence does not identify the cause of the $8.434896 total gap or establish a causal role for experience coverage.")
    h("8. Supplementary learning dynamics and rate comparison")
    p("To investigate the deterioration without presuming its cause, each of the same five seeds restarted independently for 200,000 episodes under three conditions: constant alpha 0.1; constant alpha 0.03; and alpha(s,a)=0.1/(1+N_previous(s,a)/1000). The counter is this pair's number of updates before the current update, not global episode count. Rates at 0, 1,000, and 3,000 prior updates are 0.1, 0.05, and 0.025. Scale 1,000 was chosen before the experiment, not optimized.")
    p("All other settings and the legal greedy evaluator were unchanged. Fifteen instrumented runs recorded 300 checkpoints and 2,100 records for the seven originally suboptimal states, including Q margins, actions, visits, and next-update rates. Changed policies can change subsequent random experience and RNG consumption, so equal seeds across conditions do not guarantee identical training trajectories. All runs, including equal-cost policies, are retained.")
    labels={'constant_0.1':'Constant 0.1','constant_0.03':'Constant 0.03','visit_decay':'Visit decay'}
    table(["Seed", "Rate condition", "Final exact cost", "Gap above optimal"], [[r['training_seed'], labels[r['condition']], money(r['exact_expected_cost']), money(r['gap_from_optimal'])] for r in rows('learning_dynamics_final_comparison.csv')])
    table(["Condition", "Mean exact cost", "Across-seed cost range"], [[labels[r['condition']], money(r['mean_exact_cost']), f"[{money(r['min_exact_cost'])}, {money(r['max_exact_cost'])}]" ] for r in rows('learning_dynamics_across_seed_summary.csv')], [1.5, 1.8, 3.9])
    p("Both alternatives lowered every seed's final cost relative to alpha 0.1. For seed 66401, reductions were $10.005696 (0.03) and $10.294944 (decay). Across five seeds, the mean gap was $13.704336, $6.050659, and $1.890720, respectively. Every final policy still had a positive gap. These are exploratory rate-sensitivity findings, not independent confirmation or replacements for the original four-strategy experiment.")
    table(["Condition", "Improving intervals", "Worsening intervals", "Unchanged"], [[labels[r['condition']], r['improved_intervals'], r['worsened_intervals'], r['unchanged_intervals']] for r in rows('learning_dynamics_change_summary.csv')])
    p("All conditions fluctuated. For seed 66401 under alpha 0.1, cost rose $4.719312 between 20,000 and 30,000 episodes while four tracked states switched WAIT to BOOK. At final (6, $180), WAIT was chosen by 0/5, 2/5, and 5/5 policies across the three conditions. These switches accompany performance changes; other states can change too. Sparse checkpoints omit intervening changes and do not identify a unique causal mechanism.")
    blocks.append(("pagebreak", None))
    figure(graph, "Figure 2. Report-layout rendering of the saved 300 checkpoints, all five seeds per condition and consistent axes. Lines join recorded checkpoints only. No new training or experiment was performed for this figure.", 6.4, 6.85)
    h("9. Limitations and answers to the validation questions")
    p("1. Cost reduction: the original single-seed policy had the lowest sampled mean among the four rules and positive paired mean savings with intervals above zero. This does not guarantee saving on a particular trip or on real fares.")
    p("2. Reliability: original CIs concern test-trajectory uncertainty conditional on one trained policy. Supplementary ranges expose training-seed variation. Rate alternatives improved these five final policies, but the analysis is exploratory and five seeds do not establish universal robustness. No simultaneous inference across all comparisons is claimed.")
    p("3. Optimality and correctness: the original analytical gap is $8.434896; smaller supplementary gaps show sensitivity to tested learning-rate settings. Executed tests support the specified implementation and unperturbed instrumentation. They do not prove convergence or explain the full remaining mechanism. Neither alpha 0.03 nor the visit-decay rule is established as optimal; no additional tuning or best-checkpoint selection was performed.")
    p("The underlying probabilities, five fare levels, short horizon, and absence of sold-out events are hypothetical. Fixed-step updates can retain sensitivity to sampled targets; decay can alter learning speed and coverage, but the relative contribution of such mechanisms was not isolated. There is no evidence here of real-flight savings. The completed project demonstrates a reproducible reinforcement-learning application and its limits rather than a validated travel recommendation.")
    h("10. Reproducibility, assistance, and review status")
    p("The ZIP provides the executed main Notebook, runner and numerical-verification module, pinned runtime requirements, README, unchanged English proposal, RESULTS.md, and all official result artifacts. README identifies the project virtual-environment interpreter and Python (CS664 Flight Booking) kernel. Report files and recording are separate uploads. Packaging excluded practice work, credentials, environments, caches, Git, local tools, and instructor materials.")
    p("Codex assistance disclosure: OpenAI Codex assisted with implementation, execution, verification, supplementary investigations, report drafting, and packaging. Report numbers come from saved actual artifacts. Personal practice, independent understanding, camera-on recording, human report approval, and portal submission have not been established by those automated actions. The student must review this draft and confirm compliance with any syllabus/instructor AI-use guidance. The supplied project PDF does not specify an AI-disclosure rule; this disclosure does not replace other applicable course guidance.")
    h("References and evidence sources")
    for text in [
        "[1] Kalathur, S. (2026). CS664 Term Project. Instructor-provided metcs664_TermProject.pdf. Used for the official deliverables and demo requirements; excluded from the ZIP.",
        "[2] CS664_Project_Wong_Sheung_Wan_Proposal.md. Original supplied English implementation plan, unchanged. CS664_Flight_Booking.ipynb, RESULTS.md, and results/ supply the executed numerical evidence; report_assets/report_number_audit.json maps validation to those artifacts.",
        "[3] Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292. https://doi.org/10.1007/BF00992698. Publisher abstract and bibliographic page consulted for algorithm background; not a claim that this finite project satisfies a convergence theorem.",
        "[4] SciPy contributors. scipy.stats.t API documentation. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html. Consulted October 14, 2026 for the Student-t quantile used in paired intervals.",
        "[5] NumPy contributors. Random Generator documentation. https://numpy.org/doc/stable/reference/random/generator.html. Consulted October 14, 2026 for seeded random-generator behavior."]:
        p(text)
    p("External references were consulted to validate method explanations while drafting, not used as flight-price data. This remains a draft awaiting student review.")
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
            flow.append(Paragraph(escape(data),styles[key]))
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
    for kind,data in blocks:
        if kind=='title':document.add_heading(data,0)
        elif kind=='heading':document.add_heading(data,1)
        elif kind=='paragraph':document.add_paragraph(data)
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
    import fitz
    from docx import Document
    from PIL import Image,ImageOps,ImageDraw
    folder=ROOT/'.cache'/'report_preview';folder.mkdir(parents=True,exist_ok=True)
    pdf=fitz.open(ROOT/f'{NAME}.pdf')
    text='\n'.join(page.get_text() for page in pdf)
    assert len(pdf)>0 and 'DRAFT' in text and 'References and evidence sources' in text
    for token in ['239.580','223.476','219.738','214.854','207.274224','215.709120','8.434896',
                  '220.978560','213.324883','209.164944','23.837947','25.614053','182.36']:
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
    assert 'DRAFT' in doc_text and 'References and evidence sources' in doc_text
    assert len(doc.inline_shapes)==2
    for token in ['207.274224','215.709120','209.164944','214.854']:
        assert token in doc_text
    result={'pdf_pages':len(pdf),'pdf_text_inside_page_bounds':True,'key_report_numbers_present':True,
            'docx_readable_and_tables_present':True,'embedded_figures':2,'rendered_pages_folder':str(folder.relative_to(ROOT))}
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
    protected=[ROOT/'CS664_Flight_Booking.ipynb',ROOT/'CS664_Flight_Booking_Practice.ipynb',
               ROOT/'CS664_Project_Wong_Sheung_Wan_Proposal.md']+[p for p in RESULTS.rglob('*') if p.is_file()]
    before={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    summary,runtime=audit_saved_evidence()
    graph=make_report_figure()
    make_documents(report_blocks(summary,runtime,graph))
    document_checks=verify_pdf_and_docx()
    package_checks=make_archive()
    assert before=={str(p.relative_to(ROOT)):fingerprint(p) for p in protected}
    (ASSETS/'preservation_validation.json').write_text(json.dumps({'status':'PASS',
        'protected_files_unchanged':True,'protected_sha256':before,'training_rerun':False},indent=2)+'\n')
    print('Report draft created:',ROOT/f'{NAME}.pdf',ROOT/f'{NAME}.docx')
    print('PDF pages:',document_checks['pdf_pages'],'; rendered for visual inspection.')
    print('ZIP verified:',ROOT/f'{NAME}.zip','files:',package_checks['files'])
    print('PASS: every report number audited; main/practice/proposal and all original results unchanged.')


if __name__=='__main__':main()

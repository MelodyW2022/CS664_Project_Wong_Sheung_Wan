# Flight Booking Decision Assistant

The original proposal and course PDF are preserved. The project is implemented in
`CS664_Flight_Booking.ipynb`: the specified simulator, 50,000-episode tabular
Q-learning training, greedy BOOK/WAIT predictions, four strategies on the same
10,000 fresh test trajectories, exact finite-horizon MDP references, a BOOK/WAIT
heatmap, paired 95% confidence intervals, and a complete decision trace.
No flight data, external APIs, or website are used.
The instructor's `metcs664_TermProject.pdf` supplies the official submission
requirements; the proposal Markdown is the implementation plan.

Git excludes environments, caches, credentials, and instructor course materials.
The instructor PDF remains in the local project folder and is not uploaded.
It is optional for Notebook execution after cloning this repository.

## Setup and execution

Run these commands from this project folder. The environment is already set up
here; creation and package installation are only needed on a new machine.
Python **3.13.11** was used. `requirements.txt` records the installed versions.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name cs664-flight-booking --display-name "Python (CS664 Flight Booking)"
.venv/bin/python run_notebook.py
```

To work interactively:

```sh
.venv/bin/jupyter lab CS664_Flight_Booking.ipynb
```

Select interpreter **this project's `.venv/bin/python`**, and Notebook kernel
**Python (CS664 Flight Booking)** (`cs664-flight-booking`). In VS Code, choose
the project virtual environment under Select Kernel / Python Environments.
Use **Restart Kernel and Run All Cells**, then save the Notebook. The runner
does this in a fresh kernel and saves actual outputs only after a successful run.

The supplementary investigation also trains ten independent models: five fixed
seeds at 50,000 and 200,000 episodes. Allow extra execution time. The runner
prints progress after each seed's pair and checks that the original result
files remain unchanged. The learning-dynamics follow-up adds fifteen
instrumented 200,000-episode runs (three learning-rate conditions, five seeds),
with exact evaluation every 10,000 episodes. The runner prints each completed
run; allow additional execution time. Earlier original and supplementary result
files are checked for unchanged hashes. No additional packages or commands
are required. The runner never writes the practice Notebook and permits you
to save your practice attempts while the main experiment runs.

## Official results

Start with [RESULTS.md](RESULTS.md) for actual findings and linked evidence.
The executed Notebook contains checks, observed Q-value updates, comparison,
trace, analytical references, confidence intervals, and interpretation.
Tables are CSV and figures are PNG in `results/`:

- `mean_cost_comparison.csv` and `official_comparison.md`: official means and
  mean paired savings (baseline cost minus Q-learning cost).
- `paired_purchase_costs.csv`: all 10,000 matched purchase costs.
- `test_trajectories.npy`, `q_values.npy`, `training_visit_counts.npy`: raw evidence.
- `q_learning_trace.md` and `run_summary.json`: trace, final cost, settings,
  actual updates, verification counts, and interpreter version.
- `exact_mdp_comparison.csv`, `exact_state_values.csv`: analytical optimal and
  fixed-policy expectations, averaged uniformly over initial prices.
- `policy_heatmap.png`, `learned_policy_actions.csv`: greedy BOOK/WAIT decisions.
- `paired_savings_95ci.csv`, `paired_savings_95ci.png`: mean paired savings and
  individual 95% Student-t intervals, conditional on this trained policy.
- `q_learning_trace.csv`, `q_learning_trace.png`: complete observed trace.
- `simulation_assumptions.json`, `training_parameters.csv`,
  `verification_results.csv`: model, settings, and actually executed checks.
- `supplementary_*.csv`, `supplementary_summary.json`: implementation checks,
  original coverage/action penalties, all ten training runs, paired changes,
  mean/range summaries, and analytical baseline comparisons.
- `supplementary_seed_budget_gaps.png`, `supplementary_trained_models.npz`:
  paired budget endpoints and every supplementary Q-table/visit-count array.
- `learning_dynamics_*.csv`, `learning_dynamics_config.json`,
  `learning_dynamics_summary.json`: all checkpoints, tracked states, fifteen
  final runs, paired rate comparisons, summaries, and executed checks.
- `learning_dynamics_checkpoint_gaps.png`, `learning_dynamics_final_models.npz`:
  all-seed learning dynamics and clearly labeled final Q/visit-count arrays.

Training seed **66401** and test seed **66402** are separate. Evaluation has no
exploration or Q updates. A policy sees only its current state and stops at
purchase. The official fixed threshold is **$180**.

## Submission files

Include the main Notebook with saved outputs, main runner, requirements, README,
RESULTS.md, proposal, and official `results/` in the project ZIP. Exclude
`CS664_Flight_Booking_Practice.ipynb`, `PRACTICE_GUIDE.md`, `practice_results/`,
and `verify_practice.py`; these are preparation materials. Also exclude `.venv/`,
`.jupyter/`, `.git/`, `.tools/`, caches, credentials, and instructor materials.
No submission ZIP has been created. Upload the final writeup PDF/Word file,
project ZIP, and recording separately as the instructor PDF requires.

Practice instructions and learning order are in [PRACTICE_GUIDE.md](PRACTICE_GUIDE.md).

Exact MDP references use known simulator probabilities; the learning agent
never receives them. Sampled means and paired intervals use the shared held-out
trajectories. One seeded result from hypothetical probabilities does not prove
convergence or savings on every trip. Intervals are large-sample approximations,
are not simultaneous, and do not quantify variability across training seeds.

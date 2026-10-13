# Flight Booking Decision Assistant

The original proposal and course PDF are preserved. The core is implemented in
`CS664_Flight_Booking.ipynb`: the specified simulator, 50,000-episode tabular
Q-learning training, greedy BOOK/WAIT predictions, four strategies on the same
10,000 fresh test trajectories, a mean-cost table, and a complete decision trace.
No flight data, external APIs, or website are used.

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

## Results and learning

The executed Notebook contains checks, observed Q-value updates, comparison,
trace, and interpretation. `results/` contains:

- `mean_cost_comparison.csv` and `official_comparison.md`: official means and
  mean paired savings (baseline cost minus Q-learning cost).
- `paired_purchase_costs.csv`: all 10,000 matched purchase costs.
- `test_trajectories.npy`, `q_values.npy`, `training_visit_counts.npy`: raw evidence.
- `q_learning_trace.md` and `run_summary.json`: trace, final cost, settings,
  actual updates, verification counts, and interpreter version.

Training seed **66401** and test seed **66402** are separate. Evaluation has no
exploration or Q updates. A policy sees only its current state and stops at
purchase. The official threshold is **$180**. For practice, change only
`PRACTICE_THRESHOLD` in cell **6A** and rerun that cell; the official comparison
remains intact. The default practice execution does not count as your practice.

Section **9** identifies the essential cells to personally run and explain before
recording the approximately 10-minute demo. The agent implemented and executed
the core; your personal practice and camera-on recording remain yours to do.
The separate final PDF/Word proposal/writeup and recording required by the
assignment have not been produced. Exclude `.venv/` and `.jupyter/` when packaging
the project zip; include the Notebook, runner, requirements, README, and results.

Confidence intervals, the exact MDP reference solution, and the policy heatmap
are deferred. One seeded result from hypothetical probabilities does not prove
convergence or savings on every trip, and is not real-flight advice.

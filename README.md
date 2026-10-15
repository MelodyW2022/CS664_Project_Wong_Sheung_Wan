# Flight Booking Decision Assistant

`CS664_Flight_Booking.ipynb` implements a simulated flight-booking decision
problem with tabular Q-learning. It includes the original 50,000-episode policy,
four strategies on 10,000 shared test paths, paired confidence intervals, exact
finite-horizon policy costs, a BOOK/WAIT heatmap, a complete decision trace,
and verification. Supplementary experiments compare five training seeds,
two episode budgets, and three learning-rate conditions.

## Setup and execution

Extract the ZIP and open its `CS664_Project_Wong_Sheung_Wan` folder. Run the
following commands from that folder. Python **3.13.11** was used for the saved
execution; `requirements.txt` pins the recorded environment packages.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name cs664-flight-booking --display-name "Python (CS664 Flight Booking)"
.venv/bin/python run_notebook.py
```

The runner requires the project `.venv` interpreter. It executes the main
Notebook in a fresh kernel and saves its outputs after successful execution
and verification. `project_verification.py` must remain beside the runner.
The proposal Markdown and instructor PDF are not required to run the package.
If present in a local working project, the runner checks that they stay unchanged.

For interactive execution:

```sh
.venv/bin/jupyter lab CS664_Flight_Booking.ipynb
```

Select **Python (CS664 Flight Booking)** (`cs664-flight-booking`), use
**Restart Kernel and Run All Cells**, and save the Notebook. Full execution
includes ten supplementary budget runs and fifteen instrumented learning-rate
runs; it trains models and writes result files. The runner prints supplementary
progress and checks preservation of the original numerical evidence. No network
service, API key, real airfare dataset, or additional project module is needed.
Jupyter configuration and caches are created locally during execution.

## Package contents and evidence

- `CS664_Flight_Booking.ipynb`: main Notebook with 80 saved executed code cells.
- `run_notebook.py`: fresh-kernel execution and preservation checks.
- `project_verification.py`: numerical-evidence fingerprint helper.
- `requirements.txt`: pinned Python environment packages.
- `README.md`: setup, execution, and package instructions.
- [RESULTS.md](RESULTS.md): generated findings and complete evidence index.
- `results/`: all saved tables, figures, configurations, checks, trajectories,
  Q-values, visit counts, and supplementary models.

Start with [RESULTS.md](RESULTS.md). The report's evidence paths are relative
to this folder. Original comparison data are in
[mean_cost_comparison.csv](results/mean_cost_comparison.csv),
[paired_savings_95ci.csv](results/paired_savings_95ci.csv), and
[exact_mdp_comparison.csv](results/exact_mdp_comparison.csv). Full supplementary
runs are retained in [supplementary_training_runs.csv](results/supplementary_training_runs.csv)
and [learning_dynamics_final_comparison.csv](results/learning_dynamics_final_comparison.csv).

The simulator probabilities are hypothetical. Sampled means and paired intervals
are distinct from exact model expectations; none establishes real-flight savings.

The report PDF and editable DOCX are separate files outside this ZIP. Upload the
report separately from the project ZIP. The ZIP excludes the report files,
proposal Markdown, instructor materials, practice materials, presentation scripts,
development audits, environments, caches, and credentials. Those working-project
files have not been deleted. Package verification checks saved outputs, imports,
dependencies, and file paths; it does not constitute a new full Notebook execution.

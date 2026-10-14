"""Execute the main experiment in a fresh kernel; save outputs on success."""
from pathlib import Path
import hashlib
import os
import sys

ROOT = Path(__file__).resolve().parent
if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    raise SystemExit("Run with .venv/bin/python run_notebook.py")

# Keep Jupyter configuration and runtime files inside the project.
os.environ["JUPYTER_CONFIG_DIR"] = str(ROOT / ".jupyter" / "config")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter" / "runtime")
os.environ["IPYTHONDIR"] = str(ROOT / ".jupyter" / "ipython")
os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
for variable in ("JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR"):
    Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

import nbformat
from nbclient import NotebookClient
from project_verification import numerical_evidence

originals = [ROOT / "CS664_Project_Wong_Sheung_Wan_Proposal.md"]
assignment_pdf = ROOT / "metcs664_TermProject.pdf"
if assignment_pdf.exists():
    originals.append(assignment_pdf)  # Instructor material remains local, outside Git.
practice_notebook = ROOT / "CS664_Flight_Booking_Practice.ipynb"
practice_before = (hashlib.sha256(practice_notebook.read_bytes()).hexdigest()
                   if practice_notebook.exists() else None)
before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
result_hashes = numerical_evidence(ROOT / "results")
notebook_path = ROOT / "CS664_Flight_Booking.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)

def report_training_progress(cell, cell_index, execute_reply):
    if cell.metadata.get("stage_label", "").startswith(("S3-PAIR-", "L3-RUN-")):
        for output in cell.outputs:
            if output.output_type == "stream":
                print(output.text, end="", flush=True)

client = NotebookClient(notebook, timeout=300, kernel_name="cs664-flight-booking",
                        on_cell_executed=report_training_progress,
                        allow_errors=False, resources={"metadata": {"path": str(ROOT)}})
client.execute()
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in code_cells)
assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in originals}
assert result_hashes == numerical_evidence(ROOT / "results")
nbformat.validate(notebook)
nbformat.write(notebook, notebook_path)
print(f"PASS: {len(code_cells)} code cells executed in a fresh kernel; no cell errors.")
print("Proposal and local assignment PDF hashes unchanged.")
practice_after = (hashlib.sha256(practice_notebook.read_bytes()).hexdigest()
                  if practice_notebook.exists() else None)
if practice_before == practice_after:
    print("Practice Notebook hash unchanged; this runner never writes it.")
else:
    print("Practice Notebook changed during execution; this runner did not write it.")
print("Original numerical evidence, including Q-values and visits, is unchanged.")
print(f"Saved executed Notebook: {notebook_path}")

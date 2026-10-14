"""Verify supplied practice material without filling or grading student TODOs."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
    raise SystemExit("Run with .venv/bin/python verify_practice.py")
os.chdir(ROOT)
for name, folder in {"JUPYTER_CONFIG_DIR": ".jupyter/config",
                     "JUPYTER_RUNTIME_DIR": ".jupyter/runtime",
                     "IPYTHONDIR": ".jupyter/ipython"}.items():
    os.environ[name] = str(ROOT / folder)
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")

import nbformat
from nbclient import NotebookClient

protected = [ROOT / "CS664_Flight_Booking.ipynb", ROOT / "RESULTS.md",
             ROOT / "CS664_Project_Wong_Sheung_Wan_Proposal.md"]
protected += [p for p in (ROOT / "results").rglob("*") if p.is_file()]
pdf = ROOT / "metcs664_TermProject.pdf"
if pdf.exists():
    protected.append(pdf)
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in protected}
path = ROOT / "CS664_Flight_Booking_Practice.ipynb"
original = nbformat.read(path, as_version=4)
if sys.argv[1:] == ["--part5-only"]:
    # Verify only the new interpretation material; preserve Parts 1–4 exactly.
    untouched = deepcopy([c for c in original.cells
                          if not c.metadata.get("stage_label", "").startswith("P5")])
    selected = [deepcopy(c) for c in original.cells
                if c.metadata.get("practice_role") == "setup"
                or c.metadata.get("stage_label", "").startswith("P5")]
    executed = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(original.metadata))
    NotebookClient(executed, kernel_name="cs664-flight-booking", timeout=180,
                   allow_errors=False, resources={"metadata": {"path": str(ROOT)}}).execute()
    by_id = {c.id: c for c in executed.cells}
    for cell in original.cells:
        if cell.cell_type == "code" and cell.metadata.get("stage_label", "").startswith("P5"):
            output = by_id[cell.id]
            cell.execution_count, cell.outputs = output.execution_count, output.outputs
            cell.metadata = output.metadata
    assert untouched == [c for c in original.cells
                         if not c.metadata.get("stage_label", "").startswith("P5")]
    assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in protected}
    original.metadata["practice_progress"]["part5_verification"] = "saved-evidence model answers verified; personal completion not established"
    nbformat.validate(original)
    nbformat.write(original, path)
    manifest = {"status": "PASS", "mode": "part5-only", "training_rerun": False,
                "parts1to4_cells_unchanged": True, "official_files_unchanged": True,
                "protected_file_sha256": hashes, "personal_completion": "not established"}
    (ROOT / "practice_results" / "part5_preservation_verification.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("PASS: Part 5 model answers verified using saved evidence; no training executed.")
    print("Parts 1–4 and attempts preserved; main Notebook and official result hashes unchanged.")
    raise SystemExit(0)
if sys.argv[1:]:
    raise SystemExit("Usage: .venv/bin/python verify_practice.py [--part5-only]")
executed = deepcopy(original)
checks = [c for c in original.cells if c.metadata.get("practice_role") == "student_check"]
todo_cells = [c for c in original.cells if c.metadata.get("practice_role") == "student_todo"]
source = """# Verification-only cell: never saved as a student exercise or solution.
import ast
solution_aliases = {
    'PracticeBookingEnv': SolutionBookingEnv,
    'terminal_update_practice': terminal_update_solution,
    'wait_update_practice': wait_update_solution,
    'training_action_practice': training_action_solution,
    'paired_saving_practice': paired_saving_solution,
    'mean_saving_practice': mean_saving_solution,
    'final_step_practice': final_step_solution,
}
scope = dict(globals(), **solution_aliases)
for label, source in CHECK_SOURCES:
    exec(compile(source, label, 'exec'), scope)
    solution_checks_run.append('Provided exercise check verified with separate solution: ' + label)

unfinished_functions_verified = []
calls = {
    'P1-TODO': [('PracticeBookingEnv', None)],
    'P2-TODO': [('terminal_update_practice', (0, -120, .1)),
                ('wait_update_practice', (0, 0, .1, 1, -30))],
    'P2-EPS-TODO': [('training_action_practice', ((6, 180), .05, .03, WAIT, BOOK))],
    'P3-TODO': [('paired_saving_practice', (180, 240)),
                ('mean_saving_practice', ([0, 0, -60],))],
    'P4-MDP-TODO': [('final_step_practice', ())],
}
for label, source in TODO_SOURCES:
    # Only check intentional unfinished placeholders; never grade a saved attempt.
    if not any(isinstance(node, ast.Constant) and node.value is Ellipsis
               for node in ast.walk(ast.parse(source))):
        continue
    student_scope = dict(globals())
    exec(compile(source, label, 'exec'), student_scope)
    for name, args in calls[label.split(' —', 1)[0]]:
        function = student_scope[name]
        if name == 'PracticeBookingEnv':
            env = function(FixedMoveRNG(0))
            env.reset()
            env.state = (0, 180)
            function, args = env.step, (WAIT,)
        # A partially answered function may have no placeholder: leave it alone.
        node = next(node for node in ast.parse(source).body
                    if getattr(node, 'name', '') == name)
        if not any(isinstance(n, ast.Constant) and n.value is Ellipsis
                   for n in ast.walk(node)):
            continue
        try:
            function(*args)
        except NotImplementedError:
            unfinished_functions_verified.append(name)
        else:
            raise AssertionError('Unfinished TODO should stop clearly: ' + name)
print('PASS: all supplied exercise checks verified; unfinished TODOs intentionally stop.')
verification_details = {
    'exercise_check_cells_verified': len(CHECK_SOURCES),
    'unfinished_functions_verified': unfinished_functions_verified,
    'student_answers_filled_by_verifier': False,
    'personal_completion': 'not established',
}
(PRACTICE_OUTPUT / 'exercise_verification.json').write_text(
    json.dumps(verification_details, indent=2) + '\\n')
"""
prefix = "CHECK_SOURCES = " + repr([(c.metadata["stage_label"], c.source) for c in checks])
prefix += "\nTODO_SOURCES = " + repr([(c.metadata["stage_label"], c.source) for c in todo_cells])
extra = nbformat.v4.new_code_cell(prefix + "\n" + source)
end = next(i for i, c in enumerate(executed.cells)
           if c.metadata.get("stage_label", "").startswith("P-END"))
executed.cells.insert(end, extra)
NotebookClient(executed, kernel_name="cs664-flight-booking", timeout=180,
               allow_errors=False, skip_cells_with_tag="student",
               resources={"metadata": {"path": str(ROOT)}}).execute()

by_id = {c.id: c for c in executed.cells}
for cell in original.cells:
    if cell.cell_type == "code" and "student" not in cell.metadata.get("tags", []):
        output = by_id[cell.id]
        cell.execution_count = output.execution_count
        cell.outputs = output.outputs
        cell.metadata = output.metadata
assert hashes == {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in protected}
assert [c for c in original.cells if c.metadata.get("practice_role", "").startswith("student")] == [
    c for c in nbformat.read(path, as_version=4).cells
    if c.metadata.get("practice_role", "").startswith("student")]
original.metadata["practice_progress"]["verification"] = "supplied solutions/checks executed; student completion not established"
nbformat.validate(original)
nbformat.write(original, path)
manifest = {"status": "PASS", "protected_file_sha256": hashes,
            "official_files_unchanged": True, "student_cells_unchanged": True,
            "verification": "supplied solutions and exercise checks; not personal completion"}
(ROOT / "practice_results" / "preservation_verification.json").write_text(
    json.dumps(manifest, indent=2) + "\n")
print(f"PASS: {len(checks)} exercise check cells verified with separate solutions.")
print("Main Notebook, proposal, instructor PDF, and all official results unchanged.")
print("Student TODO sources, attempts, and outputs preserved; practice outputs saved separately.")

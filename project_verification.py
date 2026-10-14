"""Portable fingerprints for numerical evidence, independent of interpreter paths."""
from pathlib import Path
import hashlib
import json

NUMERICAL_SUFFIXES = {".csv", ".npy", ".npz", ".json", ".md"}


def evidence_digest(path):
    path = Path(path)
    if path.name == "run_summary.json":
        content = json.loads(path.read_text())
        # Execution metadata changes across installations; numerical evidence does not.
        content.pop("execution", None)
        data = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    else:
        data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def numerical_evidence(results_folder):
    return {path: evidence_digest(path) for path in Path(results_folder).rglob("*")
            if path.is_file() and path.suffix in NUMERICAL_SUFFIXES
            and not path.name.startswith("learning_dynamics_")}

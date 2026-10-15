# Presentation checklist

## Before recording

- [ ] Treat script timestamps as rehearsal estimates, not measured durations. Allow time to point at each listed line, inspect rows, and navigate; do not rush or add repetitive narration.
- [ ] Camera and microphone on; test sound and readable code. Keep the camera overlay away from tables.
- [ ] Share the main Notebook and terminal. Keep the script on an unshared screen or printed page. Close practice and personal tabs; silence notifications.
- [ ] Open the project root. Select `.venv/bin/python` and kernel `Python (CS664 Flight Booking)` (`cs664-flight-booking`).
- [ ] Use readable zoom. Navigate by Notebook headings or Find for the script’s exact cell labels.
- [ ] Save any existing Notebook edits before starting. Turn off Auto Save. Do not edit, save, or execute cells in the viewer while the runner writes fresh outputs.
- [ ] Have the terminal ready. In VS Code, use Terminal → New Terminal. Avoid running a second training job.

## Full run and fresh outputs

- [ ] Start recording first. Launch the full runner from the project root by 0:20:

```sh
time .venv/bin/python run_notebook.py
```

- [ ] Explain code while it runs. Initially displayed outputs are from an earlier run.
- [ ] Look for `PASS: 80 code cells executed in a fresh kernel; no cell errors.`, unchanged numerical evidence, the saved-Notebook message, and a returned prompt without a traceback.
- [ ] At the boundary before Notebook Section 4, around 3:50, reload from disk or close/reopen the main Notebook. Do not overwrite the new file with the old viewer contents.
- [ ] Check fresh displayed values before reading the script’s numbers aloud.

Your personally observed CLI elapsed time was 2:55.26 for one run. It is not a guarantee. Start early and retain the completion/reload buffer.

If the run is still active at the planned 3:50 boundary, wait for genuine completion before showing fresh results. Explain the delay and allow an overrun, or resolve the setup issue and record another complete take. If it repeatedly cannot fit, ask the instructor how to handle the timing conflict. Do not cut training budgets, omit experiments, call a partial run complete, or assume speed-up/editing is allowed.

## Essential Notebook cells to run and explain

Use these cells for the camera-on demo. Full-run and kernel instructions are in
README.md; timed screen directions are in PRESENTATION_SCRIPT.md.

- **1A:** parameters and separate training/test seeds.
- **2A–2D:** transitions, boundaries, rewards, deadline, and termination.
- **3A–3D:** exploration, legal targets, and actual Q updates.
- **4A–4E:** shared trajectories, state-only decisions, four strategies, and costs.
- **5A:** complete trace and final purchase.
- **6A–6D, 7A–7B, 8A–8C, 9A–9C, 10A:** analytical references, heatmap, intervals, checks, and limits.
- **S3C, L4B:** supplementary budget and learning-rate summaries.

## Final recording check

- [ ] Camera stays on throughout.
- [ ] Explain environment code, the Q update, exploration, and the state-only evaluator.
- [ ] Show the whole run succeeding and the freshly reloaded outputs.
- [ ] Cover mean costs, paired intervals, exact optimality gap, heatmap/trace, and checks.
- [ ] Keep supplementary budget/rate findings to about one minute; finish with limitations.
- [ ] Check sound, readability, and approximately ten-minute duration before uploading the recording separately.

The script and this checklist stay outside the submission ZIP.

## Requirements-to-script check

- [ ] Approximately ten minutes: timestamps are rehearsal estimates, including pointing and navigation.
- [ ] Camera on: keep it visible throughout, as the instructor PDF requires.
- [ ] Explain Python code: Notebook Sections 2–4 cover the environment, Q-learning, and evaluator.
- [ ] Demo the whole project: launch the unchanged runner in Opening; wait for the final success messages, then reload from disk before Section 4. Do not claim completion while the job is active or failed.
- [ ] Show results: follow Notebook Sections 4–10 in order, including the trace, exact reference, heatmap, intervals, and verification outputs.
- [ ] Brief supplementary findings: continue downward through the two investigation sections. Short headings with only an operating cue are navigation stops, not additional explanations to read aloud.
- [ ] Upload the recording separately, as the instructor PDF specifies.

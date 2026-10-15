# Presentation checklist

## Before recording

- [ ] Treat script timestamps as rehearsal estimates, not measured durations. Allow time to point at each listed line, inspect rows, and navigate; do not rush or add repetitive narration.
- [ ] Camera and microphone on; test sound and readable code. Keep the camera overlay away from tables.
- [ ] Share the main Notebook. Keep the script on an unshared screen or printed page. Close practice and personal tabs; silence notifications.
- [ ] Open the project root. Select `.venv/bin/python` and kernel `Python (CS664 Flight Booking)` (`cs664-flight-booking`).
- [ ] Use readable zoom. Navigate by Notebook headings or Find for the script’s exact cell labels.
- [ ] Open the executed Notebook with its saved outputs visible. Keep those outputs intact while presenting.
- [ ] Use the results from the completed command-line run. Do not execute cells or start another training job during recording.

## Saved results for recording

- [ ] Present the saved results from the earlier `time .venv/bin/python run_notebook.py` run.
- [ ] State that the execution took place earlier. Explain the code and its saved outputs in the current Notebook.
- [ ] If showing terminal success messages, identify them as evidence of that earlier run.
- [ ] Check the displayed values before reading the script’s numbers aloud. Keep the Notebook open throughout the recording.

## Essential Notebook cells to show and explain

Use these cells for the camera-on walkthrough. Execution instructions are in
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
- [ ] Show the saved outputs and verification evidence from the completed run.
- [ ] Cover mean costs, paired intervals, exact optimality gap, heatmap/trace, and checks.
- [ ] Keep supplementary budget/rate findings to about one minute; finish with limitations.
- [ ] Check sound, readability, and approximately ten-minute duration before uploading the recording separately.

The script and this checklist stay outside the submission ZIP.

## Requirements-to-script check

- [ ] Approximately ten minutes: timestamps are rehearsal estimates, including pointing and navigation.
- [ ] Camera on: keep it visible throughout, as the instructor PDF requires.
- [ ] Explain Python code: Notebook Sections 2–4 cover the environment, Q-learning, and evaluator.
- [ ] Walk through the project using the completed command-line run’s saved results. Identify them as earlier results rather than a new execution during recording.
- [ ] Show results: follow Notebook Sections 4–10 in order, including the trace, exact reference, heatmap, intervals, and verification outputs.
- [ ] Brief supplementary findings: continue downward through the two investigation sections. Short headings with only an operating cue are navigation stops, not additional explanations to read aloud.
- [ ] Upload the recording separately, as the instructor PDF specifies.

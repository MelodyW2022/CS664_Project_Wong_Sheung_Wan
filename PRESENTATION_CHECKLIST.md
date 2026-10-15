# Presentation checklist

## Before recording

- [ ] Treat script timestamps as rehearsal estimates, not measured durations. Allow time to point at each listed line, inspect rows, and navigate; do not rush or add repetitive narration.
- [ ] Camera and microphone on; test sound and readable code. Keep the camera overlay away from tables.
- [ ] Share the main Notebook. Keep the script on an unshared screen or printed page. Close practice and personal tabs; silence notifications.
- [ ] Open the project root. Select `.venv/bin/python` and kernel `Python (CS664 Flight Booking)` (`cs664-flight-booking`).
- [ ] Use readable zoom. Navigate by Notebook headings or Find for the script’s exact cell labels.
- [ ] Save any existing Notebook edits before starting. Use the current Notebook for execution and viewing; let Run All finish before making edits.
- [ ] Use one Notebook run for the demonstration. Avoid starting another training job.

## Full run and fresh outputs

- [ ] Start recording first. By 0:20, restart the Notebook kernel and select Run All.

- [ ] Explain code while it runs. Distinguish earlier saved outputs from outputs produced by this run.
- [ ] Confirm Run All has finished, the kernel is idle, all 80 code cells have completed, and there are no error outputs. Check the final output.
- [ ] Before narrating fresh results in Notebook Section 4, around 3:50, confirm completion and return to that section in the current Notebook.
- [ ] Check the current displayed values before reading the script’s numbers aloud. Save the completed Notebook.

Execution time can vary. Start early and leave time to confirm completion.

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
- [ ] Show the whole run succeeding and its outputs in the current Notebook.
- [ ] Cover mean costs, paired intervals, exact optimality gap, heatmap/trace, and checks.
- [ ] Keep supplementary budget/rate findings to about one minute; finish with limitations.
- [ ] Check sound, readability, and approximately ten-minute duration before uploading the recording separately.

The script and this checklist stay outside the submission ZIP.

## Requirements-to-script check

- [ ] Approximately ten minutes: timestamps are rehearsal estimates, including pointing and navigation.
- [ ] Camera on: keep it visible throughout, as the instructor PDF requires.
- [ ] Explain Python code: Notebook Sections 2–4 cover the environment, Q-learning, and evaluator.
- [ ] Demo the whole project: restart the kernel and select Run All in Opening; confirm completion in the current Notebook before Section 4. Do not claim completion while the job is active or failed.
- [ ] Show results: follow Notebook Sections 4–10 in order, including the trace, exact reference, heatmap, intervals, and verification outputs.
- [ ] Brief supplementary findings: continue downward through the two investigation sections. Short headings with only an operating cue are navigation stops, not additional explanations to read aloud.
- [ ] Upload the recording separately, as the instructor PDF specifies.

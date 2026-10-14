# Presentation checklist

## Before recording

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
- [ ] At 4:20–4:50, reload from disk or close/reopen the main Notebook. Do not overwrite the new file with the old viewer contents.
- [ ] Check fresh displayed values before reading the script’s numbers aloud.

Your personally observed CLI elapsed time was 2:55.26 for one run. It is not a guarantee. Start early and retain the completion/reload buffer.

If the run is still active at 4:50, wait for genuine completion before showing fresh results. Explain the delay and allow an overrun, or resolve the setup issue and record another complete take. If it repeatedly cannot fit, ask the instructor how to handle the timing conflict. Do not cut training budgets, omit experiments, call a partial run complete, or assume speed-up/editing is allowed.

## Final recording check

- [ ] Camera stays on throughout.
- [ ] Explain environment code, the Q update, exploration, and the state-only evaluator.
- [ ] Show the whole run succeeding and the freshly reloaded outputs.
- [ ] Cover mean costs, paired intervals, exact optimality gap, heatmap/trace, and checks.
- [ ] Keep supplementary budget/rate findings to about one minute; finish with limitations.
- [ ] Check sound, readability, and approximately ten-minute duration before uploading the recording separately.

The script and this checklist stay outside the submission ZIP.

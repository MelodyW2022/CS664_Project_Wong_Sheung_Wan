# Submission review checklist

Target upload: **October 14, 2026, 11 PM MST**.
Official deadline provided by the student: **October 15, 2026, 3 AM MST**.

The report is a draft awaiting your review. No course-portal upload, recording,
or personal practice completion is established by the automated work.

## Separate deliverables

- [ ] Review and upload `CS664_Project_Wong_Sheung_Wan.pdf` as the writeup.
  `CS664_Project_Wong_Sheung_Wan.docx` is the editable alternative; the course
  PDF permits PDF or Word. After editing the DOCX, export a matching final PDF
  before uploading; edits do not automatically synchronize the two files.
- [ ] Upload `CS664_Project_Wong_Sheung_Wan.zip` as the project package.
- [ ] Record and upload the approximately 10-minute, camera-on presentation
  separately. Explain the key code and demonstrate full project execution.
- [ ] Confirm the English proposal's separate upload is present if required by
  the course portal; the original Markdown is also preserved inside the ZIP.

All four requested generated deliverables are in the project root. Reports
and this checklist stay outside the ZIP. The ZIP contains one top-level folder
`CS664_Project_Wong_Sheung_Wan/`. Staging is in
`submission_staging/CS664_Project_Wong_Sheung_Wan/` and is not committed.

## Report review requiring your action

- [ ] Check your name, project goal, scope, and explanation of the simulator.
  The instructor PDF supplies the rubric; the proposal supplies the plan.
- [ ] Review the original sampled costs: immediate **$239.580**, minimum-price
  **$223.476**, $180 threshold **$219.738**, original Q-learning **$214.854**.
- [ ] Review the three paired savings and 95% intervals: **$24.726**
  [$23.837947, $25.614053], **$8.622** [$7.240777, $10.003223], and
  **$4.884** [$4.029366, $5.738634]. These concern test-trajectory variation
  conditional on the original trained policy, not variation across training seeds.
- [ ] Review analytical expectations: optimum **$207.274224**, original learned
  policy **$215.709120**, gap **$8.434896**. Distinguish these from sampled costs.
- [ ] Review the actual update, heatmap, complete $360 trace, and executed checks.
- [ ] Review all ten five-seed/two-budget runs: all five final policies worsened
  at 200,000 episodes with the original fixed learning rate. More training was
  not established to improve this setup.
- [ ] Review all fifteen three-condition runs: both rate alternatives lowered
  final costs for all five seeds relative to alpha 0.1. Every final policy still
  had a positive optimality gap. This exploratory evidence does not establish
  convergence, an optimal learning rate, a unique cause, or real-flight savings.
- [ ] Review limitations, references, and accurate Codex assistance disclosure.
  Confirm any additional course guidance on AI assistance personally.
- [ ] Decide whether to retain the draft label or produce a reviewed final version.

## Checks completed by the assistant

- [x] Read the instructor requirements, proposal, executed main Notebook,
  README, RESULTS.md, and saved numerical evidence.
- [x] Recomputed four means from all 10,000 paired costs and three Student-t
  confidence intervals; checked exact costs/gap, logged update, full trace,
  supplementary tables, and their summaries against saved evidence.
- [x] Found 80 sequentially executed main code cells with no saved error outputs.
  The existing recorded full run was used; training was not rerun for packaging.
- [x] Rendered all nine PDF pages, inspected the page overview and detailed
  tables/figures, and checked text bounds. Opened the DOCX and checked tables,
  editable text, and its two embedded figures. DOCX pagination can vary by editor.
- [x] ZIP CRC check passed. Extracted the ZIP to a temporary directory and
  confirmed all 54 files match their sources, correct top-level folder, all
  linked runtime/result artifacts, required modules, and pinned dependencies.
- [x] Confirmed practice materials, report PDF/DOCX, checklist, course materials,
  environments, credentials, Git, and tools are absent from the ZIP.
- [x] Corrected the runner's interpreter-path-sensitive numerical fingerprint:
  machine-specific execution metadata no longer causes a false comparison
  failure after extraction. Checked relocation fixtures and actual numerical
  changes; compiled the runner and its required verification module.
- [x] SHA-256 preservation checks confirmed the main Notebook, practice Notebook,
  original proposal, and all 47 official result files were unchanged.

Audit records are in `report_assets/` in the source repository, outside the ZIP.
The recorded first-cell-busy to last-cell-idle interval was **182.36 seconds**;
this excludes kernel startup and runner postchecks. Full CLI wall time was not
measured. Full execution includes supplementary training; no fixed runtime is
promised for your machine. No full training run was performed in the extracted
package solely for packaging; compilation, references, fixtures, and saved
execution evidence were checked.

## Recording and upload actions still pending

- [ ] Personally run and explain the essential main-project cells. Automated
  execution of supplied solutions does not mean your practice is complete.
- [ ] Record goal, environment, key code, execution, results, and limitations.
  Keep the core project central and allow about one minute for supplements.
- [ ] Verify camera, sound, code readability, and video duration; demonstrate
  full execution, allowing for the supplementary runs.
- [ ] Review final filenames and portal requirements, upload the separate files,
  and verify successful upload/receipt before the target time.

No unresolved packaging failure was found. Student review, recording, any final
editor/export check after edits, and course-portal submission remain pending.

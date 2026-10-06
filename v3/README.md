# V3 result distribution — deployment candidate

The archive contains completed historical experiments. "Complete" in the two
filenames describes the packaged results, not approval of this deployment branch.
Current sequence: source-text review → deployment branch → execution verification
→ main promotion → public presentation. This source revision awaits step 2.

## Colab: exactly two user files

Put these files together in `MyDrive/PINN` and Run all:

1. `Online_PINN_PDE_Framework_V3_Complete.ipynb`
2. `Online_PINN_PDE_Framework_V3_Complete_Results.zip`

Default `ACTION="results"` reads the 65 completed experiments across nine
equations without importing TensorFlow, running an LLM, or starting training.
No Stage7 ZIP or original model ZIP needs to be selected separately.

The notebook and `.py` distribution have the same readable source. Lazy runtime
factories are ordinary Python, not base64, escaped source strings, or executable
code extracted from results. The original inputs are not overwritten.
After editing `v3/*.py`, run `python tools/sync_v3_notebook.py` to synchronize the
notebook. Colab settings/entry cells live in `v3/colab_config.py` and
`v3/colab_entry.py`; no encoded numerical-runtime payload is generated.

## Actions

| ACTION | Purpose | Runtime |
|---|---|---|
| results | Saved comparison and evidence | NumPy only; CPU |
| predict | Selected saved model / explicit points | TensorFlow |
| agent | Bounded result tools, optional Qwen | FastAPI; optional PyTorch/Transformers |
| evaluate | 30 questions + 15 independently phrased questions | Same as Agent + HTTPX |
| train | Explicit new experiment or Adam checkpoint continuation | TensorFlow + SciPy |

LLM choice preserves V1/V2: `Qwen/Qwen3.5-0.8B`. Deterministic tools supply
numbers. Qwen advises the intent; the grounded controller builds allowlisted calls.
Planner agreement and controller fallbacks are recorded separately. No shell, deletion, public server,
or automatic retraining. Configuration proposals require user confirmation.
Actual LLM results and non-LLM checks are reported separately, including failures.

Dependencies are loaded only for the selected action. Use Colab's TensorFlow
instead of forcibly replacing it. For optional services install `fastapi httpx`;
for LLM and hybrid E5 retrieval `torch transformers>=5`.
LLM weights download on first use, not on simple result review.

Hybrid retrieval restores V1/V2's multilingual E5 + BM25, semantic/keyword
weights 0.6/0.4, RRF k=60 and document-level coverage. Normalized exact inner
products equal FAISS IndexFlatIP scores; the lightweight adapter uses NumPy
without requiring FAISS binaries. E5 mean pooling and query/passage prefixes
are preserved. `DENSE_RAG=True` is the Agent/evaluation default; False explicitly
selects BM25 only. Default results viewing does not build an index or load E5.
Compact indexed evidence links to complete source histories; original histories
remain in the ZIP. Dense numeric embeddings are cached by corpus hash.

## Additional training

Select `ACTION="train"`, equation, `EXECUTION="new"` or `"resume"`.
`TRAIN_OVERRIDES` is an editable dictionary. Full defaults print before execution.
Keep `TRAINING_ENABLED=False` to inspect settings without updates. After reviewing
the settings, set it to True and Run all. There is no second RUN phrase prompt.

Example: `{"arms": ["ff"], "cap": 100000, "base_lr": 0.002}`.
The total cap includes already completed Adam steps when resuming.

For historical completed checkpoints leave `RESUME_ZIP_FILENAME=""`.
For a newly created additional-training ZIP specify its filename (not a full path).
Resume restores Adam, both optimizer states where applicable, SA and sampling
cycle. It does NOT pretend the L-BFGS-selected model has the Adam optimizer state.
Changed architecture, sampling, seed or optimizer schedule requires a new run.
Target/total cap can change with the change history retained.

Candidate adapters expose width/depth, FF banks, learning rates, all training
sample counts, LHS period, SA, stop policy and L-BFGS options. Evaluation points
are independent and fixed. Wave preserves its historical product FF architecture,
calibration, sample counts and SA implementation; unsupported changes are rejected
with the precise names instead of being ignored.

Each run writes one rolling ZIP after preparation and completed trial boundaries.
No partial backup ZIP sequence. A runtime disconnect may lose the unfinished trial.
Input results remain unchanged; final output includes settings and runtime records.

## Evidence limitations

- Wave and Heat: 3 seeds each. Other seven equations: 1 seed each.
- Preserve historical Wave 5% stopping; do not relabel it as a prospective 10% run.
- Adam and selected postprocessing are separate; no equal-budget superiority claim.
- Do not compare raw residual units across equations. Forward coefficients are N/A.
- 10% is a research milestone, not industrial certification. BC/IC and worst-point
  acceptance remain deferred to V4.
- Version changes are recorded; cross-version identical training paths are not claimed.

## GitHub

Target repository: https://github.com/danny97041/online-pinn-pde-framework

Publish the readable `v3/` source and notebook. Put the completed result ZIP in a
GitHub Release so users download exactly the notebook and that ZIP. Do not commit
the LLM cache, intermediate source ZIPs, credentials, or test dependency folders.

The local verification records are inside `validation/` of the result ZIP.
Unavailable or failed actual LLM tests must not be described as passed.

Local saved-model inference and reference-error reconstruction are checked
separately from checkpoint continuation. On Windows, TensorFlow checkpoint writes
under a Unicode workspace path could not be tested successfully. In Colab use
`ACTION="evaluate"`, `USE_LLM=False`, `CHECK_TRAINING=True` for six disposable
new/resumed trial Adam updates plus separate wiring probes. This is a behavior
check, not a benchmark rerun.

When `CHECK_TRAINING=True`, the evaluation output records training activity and
combines the behavior-check outcome with the service checks. Historical passing
records do not substitute for execution verification of a modified source revision.

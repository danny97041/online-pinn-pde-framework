# V3 deployment gates

This is a deployment candidate, not a main release or physics certification.

| Step | Scope | Status |
| --- | --- | --- |
| 0 | Readable Python text, syntax, numerical AST preservation, synchronized notebook | Prepared locally |
| 1 | Publish `codex/v3-complete-distribution`; preserve V1/V2 and main | Deployment branch only |
| 2 | Execute this exact source in Colab and inspect the checked result ZIP | Pending |
| 3 | Promote the verified commit to main | Pending; requires approval |
| 4 | Update the public project overview and presentation | Pending |

## Two-file results path

Keep these together in `MyDrive/PINN`:

- `Online_PINN_PDE_Framework_V3_Complete.ipynb`
- `Online_PINN_PDE_Framework_V3_Complete_Results.zip`

Use the default `ACTION="results"`. This path must not install TensorFlow, load
Qwen, initialize a GPU, request an original source ZIP, or start training.
The result ZIP is a separate distribution asset, not a Git source blob.

## Step 2 checks

1. Results-only Run all, ZIP hashes, 65 trials and nine equations.
2. Saved model prediction and fixed-grid error reconstruction; no exact TF version pin.
3. Natural-language tests: ten easy, ten medium, ten hard, plus 15 alternate phrasings.
4. Separate deterministic/API checks from real Qwen + E5/BM25 checks. Record
   planner agreement and controller fallback; do not call controller success
   standalone LLM success.
5. Optional `ACTION="evaluate"`, `USE_LLM=False`, `DENSE_RAG=False`,
   `CHECK_TRAINING=True`: six disposable trial Adam updates plus separate wiring
   probes test new/resumed training. This is actual limited training, not a benchmark.
6. Check opt-in selected-equation settings, completed-boundary ZIP saving and no partial ZIPs.

`Online_PINN_PDE_Framework_V3_Checked_Results.zip` contains the requested
evaluation evidence; extra console logs are unnecessary unless a cell fails.
Historical evidence inside the complete ZIP is preserved but cannot qualify
this revised source by itself. BC/IC acceptance and high-precision local physics
remain deferred; V3's ten-percent field milestone is not industrial approval.

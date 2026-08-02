# Documentation Audit

Audit date: 2026-08-02
Audited revision: `dcc16c7` (tag `v1.0-baseline`, working tree clean)
Verified facts: `python -m pytest tests/ -q` → **203 passed**.

This document lists every documentation gap found in the repository, the action
taken, and where to find the resolution. It is a living record: when you change
the system, update the affected documents listed below.

---

## 1. Missing documents

All of the following were absent and have been created at the repository root.

| Document | Why it was needed | Status |
|---|---|---|
| `SYSTEM_OVERVIEW.md` | No single entry point explaining what the system is and how the two phases fit together. | Created |
| `RESEARCH_PHILOSOPHY.md` | The *why* (no indicators, causality, reproducibility, embeddings-not-trading) was scattered across plans, never stated as a coherent stance. | Created |
| `TRAINING_GUIDE.md` | The exact training procedure (prereqs → smoke → full run → resume → checkpoint lifecycle) was implied by code, never written down. | Created |
| `EVALUATION_GUIDE.md` | Five embedding-eval modules + the baselines harness had no usage or interpretation guide. | Created |
| `CHECKPOINT_FORMAT.md` | Checkpoint files were only documented via a plan and a test; no format spec existed. | Created |
| `CONFIG_REFERENCE.md` | Every config key was undocumented (meaning / default / allowed values / effect). | Created |
| `DATA_FLOW.md` | No end-to-end description of data movement from Binance archive to model tensor. | Created |
| `MODEL_CARD.md` | The teacher model had no model card (deliverable promised by `Phase2_Implementation_Plan_v1.md` §12). | Created |
| `REPRODUCIBILITY.md` | Fingerprint / manifest / snapshot / seeds / git provenance were never explained together. | Created |
| `GPU_TRAINING_GUIDE.md` | GPU/Colab training, Drive setup, resume, and failure recovery were only in a notebook. | Created |
| `COLAB_GUIDE.md` | The notebook existed but had no prose walkthrough or dependency on Drive layout. | Created |
| `TROUBLESHOOTING.md` | No failure catalog existed. | Created |
| `MODULE_REFERENCE.md` | Required by the "every module" mandate (Purpose / Inputs / Outputs / Assumptions / Side effects / Failure modes / Dependencies / Configuration / Tests). | Created |
| `DOCUMENTATION_AUDIT.md` | This file. | Created |

## 2. Stale documents

| Document | Issue | Resolution |
|---|---|---|
| `walkthrough.md` | Claims "98 tests pass"; suite is now 203. Also a point-in-time record of a completed remediation ("Phase 1: Remediation Complete", scorecard). | Marked as historical record. Numbers reflect the state at the time of writing, not today. Header note added. |
| `Phase1_Implementation_Plan_v4.md` | A design plan (v4). Structure is historical; e.g. it lists `src/training/benchmark.py`, `reports.py` at root, `src/data/...` — most is accurate but file layout has drifted (e.g. `main.py` unified CLI vs per-module entries). | Kept as the Phase 1 design record; `DATA_FLOW.md` and `CONFIG_REFERENCE.md` document current reality. |
| `Phase2_Implementation_Plan_v1.md` | Implementation divergences: (a) §6.1 says stride is a `windows[::k]` index subsample of the stride-1 window set; the implementation builds windows *directly at the requested stride* in `WindowingEngine` (`trainer.py::_build_windows_for_symbol`); (b) §6.4 says run_id = "UTC timestamp + short config hash"; implementation is `YYYYMMDD_HHMMSS_{smoke|full}`; (c) §6.4 says manifest includes snapshot ID + dataset fingerprint; `manifest.json` stores run_id/configs/git_commit/checkpoints only; (d) §6.7 shows `num_workers: 4`; `trainer_v1.yaml` ships `num_workers: 2`. | Kept as the Phase 2 design record. `CHECKPOINT_FORMAT.md`, `TRAINING_GUIDE.md`, and `CONFIG_REFERENCE.md` document implemented behavior as authoritative. |
| `README.md` | Mostly current but had no documentation index, no pointer to the 12 new guides, and a few stale details (e.g. project-structure tree omitted the Colab notebook and new docs). | Updated: added a Documentation index section, corrected the tree, linked the new guides. |

## 3. Incorrect documents (factual errors found)

| Location | Error | Verification |
|---|---|---|
| `README.md` "Project Structure" | The tree described `storage/` and `evaluation/` layout incompletely and omitted `colab_training.ipynb`. | Corrected against the live directory tree. |
| `README.md` Phase 2 bullets | Correct in substance, but "seen-once" and test-split semantics were unqualified; now clarified in `TRAINING_GUIDE.md` and `EVALUATION_GUIDE.md`. | Matches `training_manifest_v1.json`. |
| `Phase2_Implementation_Plan_v1.md` §6.1 | "never passed to `WindowingEngine`" is false: `train_window_stride` is passed into `WindowingEngine` as `stride` in `trainer.py`. | `src/training/trainer.py:64-72`. |
| `Phase2_Implementation_Plan_v1.md` §6.4 | Run-id format and manifest fields as noted above. | `src/training/train_teacher.py:63`, `src/training/checkpoint.py:101-108`. |
| Root `DATASET.md` | Correct content; duplicate generator `datacard_builder.build_global_datacard()` and the hand-written file can drift. No action needed today; noted as a maintenance risk. | — |
| `README.md` test count | "203 tests, all passing" — **verified true** (203 passed). | `pytest tests/ -q` |

No *code* bugs were found during the audit; this is a documentation audit only.

## 4. Missing diagrams

| Missing | Where it should live | Resolution |
|---|---|---|
| End-to-end system / data-flow diagram | `SYSTEM_OVERVIEW.md`, `DATA_FLOW.md` | Added Mermaid diagrams. |
| Training-pipeline diagram (data → window → mask → encoder → loss → checkpoint) | `TRAINING_GUIDE.md` | Added Mermaid diagram. |
| Checkpoint/resume lifecycle diagram | `CHECKPOINT_FORMAT.md` | Added Mermaid sequence. |
| Repository map | `SYSTEM_OVERVIEW.md` | Added tree + doc map. |

## 5. Missing examples

| Missing | Resolution |
|---|---|
| End-to-end worked example (download → … → snapshot → train → eval) | Added to `TRAINING_GUIDE.md` and `DATA_FLOW.md`. |
| Example configs for a "returns-style" run and a span-mask run | Added to `TRAINING_GUIDE.md` (reference existing configs). |
| Example checkpoint directory listing | Added to `CHECKPOINT_FORMAT.md`. |
| Example evaluation JSON output walk-through | Added to `EVALUATION_GUIDE.md`. |

## 6. Missing CLI examples

| Missing | Resolution |
|---|---|
| `main.py` subcommand examples (download/convert/resample/align/build-lake/validate/quality-report/snapshot/report/benchmark) | Consolidated in `DATA_FLOW.md` (stage-by-stage) and `CONFIG_REFERENCE.md`. |
| `train_teacher` CLI (smoke / full / resume / returns variant) | `TRAINING_GUIDE.md`. |
| All five embedding eval CLIs | `EVALUATION_GUIDE.md`. |
| Baselines harness CLI | `EVALUATION_GUIDE.md`. |
| Benchmark CLI | `TRAINING_GUIDE.md` (appendix). |
| Colab cells | `COLAB_GUIDE.md` cell-by-cell. |

## 7. Missing configuration explanations

Every key in the 13 config files was undocumented. All are now covered in
`CONFIG_REFERENCE.md` (meaning / default / allowed values / effect on training),
including: `download`, `storage`, `validation`, `dataset`,
`modalities_v1`, `alignment_v1`, `market_state_schema_v1.json`,
`windowing_v1`, `model_v1`, `model_returns_v1`, `optimizer_v1`,
`trainer_v1`, `trainer_returns_v1`.

---

## 8. Module coverage

`MODULE_REFERENCE.md` documents all 50 source modules (plus the `main.py` and
`train_teacher.py` entry points) with the required nine attributes (Purpose,
Inputs, Outputs, Assumptions, Side effects, Failure modes, Dependencies,
Configuration, Tests). Verified test-file mapping included.

## 9. How the documents fit together

```
README.md                     — entry point, quickstart, test suite
├── SYSTEM_OVERVIEW.md        — what this is, layered pipeline, repo map
├── RESEARCH_PHILOSOPHY.md    — why the design decisions exist
├── DATA_FLOW.md              — data movement from Binance to tensors
├── CONFIG_REFERENCE.md       — every config key
├── MODULE_REFERENCE.md       — every source module
├── TRAINING_GUIDE.md         — how to train + resume
├── EVALUATION_GUIDE.md       — how to evaluate embeddings
├── CHECKPOINT_FORMAT.md      — on-disk run format
├── MODEL_CARD.md             — what the model is / isn't
├── RESEARCH_BASELINE.md      — frozen ground truth + success criteria
├── REPRODUCIBILITY.md        — how results are locked
├── GPU_TRAINING_GUIDE.md     — GPU execution + recovery
├── COLAB_GUIDE.md            — notebook walkthrough
├── TROUBLESHOOTING.md        — failure catalog
└── ARCHITECTURE.md           — principles + 4 contracts (existing)
```

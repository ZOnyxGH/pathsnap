# Teammate A handoff: metrics, model training, HPO

Read `docs/PLAN.md` sections 1, 4, 5, 6.2-6.4, 8, 10 once. This doc is the condensed version for your part.

## 1. Project in 5 lines

- Input per trip: a GPS track (`lon lat t_seconds`, ~1 Hz) plus that trip's own road network (`.nodes`, directed `.arcs`). Training trips also have `.route` (ground-truth arc sequence).
- For each GPS point we generate up to 16 candidate arcs within 50 m and compute 20 features per (point, candidate) pair (Merull, `src/features.py`).
- **Your job:** a binary classifier that scores P(this candidate is the true arc). The argmax per point is the classifier-only prediction; the probabilities feed B's HMM.
- Compare >= 6 model families with the same GroupKFold folds, pick a winner, tune it with Optuna TPE, save `artifacts/models/final.joblib`.
- Baselines to beat (Top-1 on labelled points): nearest arc 55.1 % train / 53.7 % student_test; nearest + heading rule 83.1 % / 84.2 %.

## 2. Your deliverables

### `src/metrics.py` (PLAN 6.4, fixed)

```python
def point_metrics(cands: pd.DataFrame, p: np.ndarray) -> dict
    # keys: top1, macro_f1, log_loss, miss_rate, n_points
def point_top1_from_assignment(label_arc: np.ndarray, pred_arc: np.ndarray) -> float
    # used to score HMM output with the same definition (ignores label_arc == -1)
```

Definitions (PLAN 5, use exactly these, B and C score with your functions):
- **Top-1**: over labelled points (`label_arc >= 0`), the arc with max `p` within the point's rows equals `label_arc`. If the true arc is not among the candidates (a miss), the point counts as wrong.
- **Macro F1**: candidate-level binary; `pred = 1` iff the row is its point's argmax; `f1_score(label, pred, average="macro")` over classes {0,1}. Only rows of labelled points.
- **Log loss**: candidate-level `log_loss(label, clip(p, 1e-6, 1-1e-6), labels=[0,1])`, labelled points only.
- **Miss rate**: fraction of labelled points with no row where `label == 1`.
- `n_points`: number of labelled points scored.
- `point_top1_from_assignment`: `mask = label_arc >= 0; mean(pred_arc[mask] == label_arc[mask])`. `pred_arc == -1` (no candidates) is wrong.

Implementation hint: rows are sorted by `(trip_id, pt_idx, dist)`; get the argmax per point with `df.assign(p=p).groupby(["trip_id","pt_idx"])["p"].idxmax()`, or faster with a numpy group key. Keep it vectorised: the train table is ~1.5M rows.

### `src/train.py` (CLI)

Must contain (PLAN 6.3, fixed):

```python
def predict_proba(bundle: dict, cands: pd.DataFrame) -> np.ndarray   # P(label=1) per row
    # return bundle["model"].predict_proba(cands[bundle["features"]].to_numpy(np.float32))[:, 1]
```

Suggested (your call, keep it small):

```python
MODELS: dict[str, Callable[[], estimator]]   # "lr","nb","rf","xgb","lgbm","mlp",("knn")
def make_folds() -> pd.DataFrame              # trip_id, fold; writes artifacts/results/folds.csv
def cross_validate(name, df, folds) -> pd.DataFrame
def main()                                    # python -m src.train [--models lr,nb,...] [--subsample 300000]
```

Models (PLAN 5, all trained on `FEATURES` from `src.features`, same folds):

| key | Model | Config |
|---|---|---|
| `lr` | `Pipeline([StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)])` | linear |
| `nb` | `GaussianNB()` | probabilistic baseline |
| `rf` | `RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42)` | bagging |
| `xgb` | `XGBClassifier(tree_method="hist", n_estimators=400, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)` | boosting |
| `lgbm` | `LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)` | boosting, 2nd impl |
| `mlp` | `Pipeline([StandardScaler(), MLPClassifier((64, 32), early_stopping=True, random_state=42)])` | neural |
| `knn` (optional) | `Pipeline([StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance")])` | instance-based, subsampled |

Record initial hyperparameters exactly as used: the report asks for them per model.

Outputs (all committed):

| File | Content |
|---|---|
| `artifacts/results/folds.csv` | `trip_id, fold`. `GroupKFold(n_splits=5)` over the 78 clean train trips sorted by trip_id. **Every model, HPO and B's HMM evaluation reuse this file.** Write it first and commit it early |
| `artifacts/results/cv_results.csv` | one row per (model, fold): `model, fold, top1, macro_f1, log_loss, miss_rate, n_points, train_s, infer_ms_per_1k, n_train_rows` |
| `artifacts/results/model_summary.csv` | one row per model: CV mean/std of each metric, student_test `top1, macro_f1, log_loss, precision, recall`, `train_s` (full refit), `infer_ms_per_trip` |
| `artifacts/results/student_test_results.csv` | per model per student_test trip: `model, trip_id, top1, macro_f1, log_loss` |
| `artifacts/plots/cv_macro_f1.png` | bar (mean, std error bars) or box over folds, all models |
| `artifacts/plots/cv_top1.png` | same for Top-1, with horizontal lines at 55.1 % and 83.1 % baselines |
| `artifacts/plots/student_test_top1_f1.png` | grouped bars: student_test Top-1 and Macro F1 per model |
| `artifacts/plots/log_loss.png` | CV + student_test log loss per model |
| `artifacts/plots/confusion_<model>.png` | 2x2 candidate-level confusion (label vs argmax pred), student_test, row-normalised + counts |
| `artifacts/plots/timing.png` | train s and infer ms per model (log scale) |
| `artifacts/plots/mlp_loss_curve.png` | `loss_curve_` and `validation_scores_` from the MLP. Optionally boosting eval curves (`evals_result_`) for LGBM/XGB on one fold. **No curves for LR/NB/RF/kNN**: they have no epoch history, and fabricated curves are penalised |
| `cache/models/<model>.joblib` | (not committed) each model refit on all 78 trips, in the 6.3 bundle format, so B can compute route continuity per classifier |
| `cache/oof_<model>.pkl` | (not committed) out-of-fold `p` for every train row: `trip_id, pt_idx, arc_id, p`. B needs at least the winner's OOF to run the HMM on CV folds without leakage |

### `src/hpo.py` (CLI)

`python -m src.hpo --model lgbm --trials 40` (25 if short on time).

- `optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=MedianPruner(n_warmup_steps=2))`.
- Objective = mean CV Top-1 over the 5 folds from `folds.csv`. After each fold call `trial.report(running_mean, fold)` and `if trial.should_prune(): raise optuna.TrialPruned()`. Also store macro_f1/log_loss per trial with `trial.set_user_attr`.
- Search space (PLAN 5), LightGBM: `n_estimators` 200-1500, `learning_rate` 0.01-0.3 log, `num_leaves` 15-255 log, `max_depth` -1 or 4-12, `min_child_samples` 10-200 log, `subsample` 0.5-1 with `subsample_freq=1`, `colsample_bytree` 0.5-1, `reg_lambda` 1e-3-10 log, `reg_alpha` 1e-3-10 log. XGBoost: same but `max_depth` 3-10 and `min_child_weight` 1-20 instead of leaves/min_child_samples.
- `study.trials_dataframe().to_csv("artifacts/hpo/trials.csv")` (write after every trial with a callback so a crash doesn't lose work).
- Refit best params on all 78 clean trips, evaluate on student_test, save:

```python
# artifacts/models/final.joblib  (PLAN 6.3, fixed)
{"name": "lgbm", "model": fitted_estimator, "features": FEATURES, "params": {...}, "cv_top1": float}
```

- `artifacts/results/hpo_comparison.csv`: rows `untuned`, `tuned`; columns `cv_top1, cv_macro_f1, cv_log_loss, st_top1, st_macro_f1, st_log_loss, infer_ms_per_trip, train_s`. Same folds, same partitions for both.
- Plots: `artifacts/plots/hpo_history.png` (objective per trial + best-so-far), `artifacts/plots/hpo_param_importance.png` (`optuna.importance.get_param_importances`), `artifacts/plots/hpo_tuned_vs_untuned.png`.
- Write `final.joblib` with the untuned winner as soon as CV picks it (T+2.5 h) so B and C can start; overwrite with the tuned one after HPO. Keep it under 50 MB (check `num_leaves * n_estimators`; RF with 300 deep trees would be too big, which is one more reason the winner is expected to be boosting).

## 3. Inputs you consume

**Candidate table** (PLAN 6.2), from `src.features.load_dataset(split)` -> reads `cache/{split}_candidates.pkl`, builds it if missing:

| column | dtype | note |
|---|---|---|
| `trip_id` | str | |
| `pt_idx` | int32 | row index in `.track` |
| `arc_id` | int64 | |
| 20 feature columns | float32 | `from src.features import FEATURES`, in that order: `dist, dist_rank, dist_gap, dist_ratio, proj_frac, edge_proj, arc_len, bearing_diff, cos_bearing, heading_ok, speed, dt_prev, dt_next, n_cand, has_twin, out_deg, in_deg, prev_link, next_link, fallback` (check the list in code, it is the source of truth) |
| `label` | int8 | 1 iff `arc_id == label_arc` |
| `label_arc` | int64 | point's true arc, -1 = unlabelled (repeated on each row) |

Rows sorted by `(trip_id, pt_idx, dist)`. Points with no candidates have no rows.

Rules for using it:
- Drop trips in `src.data.EXCLUDED_TRIPS` (`00000033`, `00000054`) before anything (folds, training, CV scoring). `build_dataset("train")` already skips them; filter again anyway, it is cheap and protects against an old cache.
- Train only on rows of labelled points (`label_arc >= 0`). Unlabelled points are also excluded from all point metrics.
- Never use `trip_id`, `pt_idx`, `arc_id`, `label_arc` as features. Use `bundle["features"]` / `FEATURES` only.
- Class balance: ~1 positive per ~7 rows. LR uses `class_weight="balanced"`; trees are fine without it; do not resample in a way that breaks per-point groups.
- Subsampling for MLP/kNN (and LR if slow): sample **whole trips** until ~300k rows, inside the training fold only. Score on the full validation fold.

**What exists when:**
- Now: write `metrics.py` and the `train.py` loop against a synthetic frame:

```python
import numpy as np, pandas as pd
from src.features import FEATURES
rng = np.random.default_rng(0)
n_pts, k = 3000, 6
df = pd.DataFrame(rng.random((n_pts * k, len(FEATURES)), dtype=np.float32), columns=FEATURES)
df["trip_id"] = np.repeat([f"{i:08d}" for i in range(n_pts * k // 600)], 600)[: n_pts * k]
df["pt_idx"] = np.repeat(np.arange(n_pts), k).astype("int32")
df["arc_id"] = rng.integers(0, 5000, n_pts * k)
true = rng.integers(0, k, n_pts)
df["label"] = (np.tile(np.arange(k), n_pts) == np.repeat(true, k)).astype("int8")
df["label_arc"] = np.repeat(df["arc_id"].to_numpy().reshape(n_pts, k)[np.arange(n_pts), true], k)
df.loc[df["label"] == 1, "dist"] *= 0.3          # make it learnable
```

- T+45 min: `artifacts/sample/train_sample_candidates.csv` (3 real trips, same columns). Load with `pd.read_csv(..., dtype={"trip_id": str})`. Note: trip_id must stay a zero-padded string.
- T+1.5 h: full cache. Run `python -m src.features train` and `python -m src.features student_test` once on your laptop (minutes, not seconds).

## 4. Step-by-step tasks

| By | Task |
|---|---|
| T+0 | `pip install -r requirements.txt`; check `import lightgbm, xgboost, optuna` |
| T+20 min | `metrics.py` done + a tiny `if __name__ == "__main__"` assert check on a hand-made 2-point frame (one hit, one miss -> top1 0.5, miss_rate 0.5) |
| T+45 min | `train.py` skeleton runs all 6 models on the synthetic frame; `make_folds()` written |
| T+1 h | Same loop on the 3-trip sample CSV (use 3 folds there, just to smoke-test) |
| T+1.5 h | Full cache built. Commit `folds.csv`. Start CV: fast models first (NB, LR, LGBM, XGB), then RF, then MLP/kNN on subsample |
| T+2 h | `cv_results.csv`, student_test eval, all plots, OOF for every model to `cache/` |
| T+2.5 h | Pick winner (best mean CV Top-1, log loss breaks ties). Save untuned `final.joblib`, tell B and C. Start HPO |
| T+3 h | HPO done (or cut to 25 trials), tuned `final.joblib`, `trials.csv`, `hpo_comparison.csv`, HPO plots committed. Push, PR |
| after | Draft your report sections (section 5 below) with real numbers from the CSVs |

Timing: measure `train_s` with `time.perf_counter()` around `fit`, inference as ms per 1k rows and ms per trip (mean rows per trip x per-row cost, or time one real trip). Same machine for all models; note the CPU in the report.

## 5. Report sections you own

### 3.1 Overall model comparison table

Columns required by the brief: `Model | Algorithm family | Key configuration | CV Macro F1 | Top-1 accuracy | Route continuity | Student-test Macro F1 | Log loss | Inference time`. PLAN 10 adds `miss rate | route F1 | student-test Top-1 | train s`. Rows: 6 classifiers (classifier-only argmax), tuned winner, winner + HMM, Gaussian HMM (lam=0), two naive baselines. Route continuity/F1 for classifier-only rows come from B (`evaluate.py --bundle cache/models/<model>.joblib --no-hmm`); give B your bundles. Write `N/A` + reason where a metric doesn't apply (e.g. log loss for the nearest-arc baseline).

### 3.2 Compulsory questions, one subsection per model (numbering as in the brief)

- A1. Why did you consider this algorithm suitable for the PathSnap problem?
- A2. What are the major strengths and limitations of this algorithm for this task?
- B5. What were the important hyperparameters and their initial values?
- B6. What loss function or optimisation objective was used?
- C7. What was the model's performance on the student-testing dataset?
- C8. Observation-level Top-1 accuracy, Macro F1 and log loss on the student-testing dataset?
- C9. Did you observe overfitting or underfitting? What evidence? (Use train-fold vs validation-fold metrics, fold std, student_test vs CV gap, MLP loss curves.)

Answers must cite your numbers, not textbook text. Useful hooks: NB assumes feature independence but `dist`, `dist_gap`, `dist_ratio`, `dist_rank` are strongly correlated; LR can't model the `bearing_diff x heading_ok` interaction; trees handle the `heading_ok` gate natively; how well each separates reverse twins (look at points where `has_twin == 1`).

Also explain: primary metric = point Top-1 (it is what the evaluator's accuracy measures, and per-point argmax is what decoding uses); log loss matters because the HMM consumes probabilities.

### 3.3 Model visualisations (minimum)

- CV Macro F1 comparison across all models
- Student-test Top-1 and Macro F1 comparison
- Log loss comparison
- Confusion matrices (optional but cheap, do them)
- Training/validation loss plots only for models with epoch-wise histories (MLP; boosting iteration curves allowed)
- A plot that justifies model selection (CV Top-1 with fold spread + baselines)

### 4. Bayesian hyperparameter optimisation (answer all 6)

1. Which hyperparameters were included in the search?
2. What ranges or distributions were considered?
3. What objective function was optimised?
4. How many optimisation trials were performed? (and how many pruned)
5. What were the best hyperparameter values obtained?
6. Did the tuned model outperform the untuned model?

Compare untuned vs tuned on: CV Macro F1, student-test Top-1, student-test Macro F1, log loss, inference time, same metrics and partitions. Report the computational cost (total HPO wall time, time per trial).

## 6. Definition of done + verification

- [ ] `python -c "from src.metrics import point_metrics, point_top1_from_assignment"` works; self-check asserts pass (`python -m src.metrics`).
- [ ] `artifacts/results/folds.csv` has 78 rows, 5 folds, no 33/54, each trip in exactly one fold.
- [ ] `python -m src.train` reproduces `cv_results.csv` (seeds fixed at 42) with >= 6 models x 5 folds.
- [ ] Winner's CV Top-1 > 84 % (beats the heading-rule baseline). If not, tell Merull immediately.
- [ ] `python -m src.hpo --model <winner> --trials 40` writes `artifacts/hpo/trials.csv` and `final.joblib`.
- [ ] Bundle check:
  ```bash
  python -c "import joblib; from src.features import load_dataset; from src.train import predict_proba; from src.metrics import point_metrics; b=joblib.load('artifacts/models/final.joblib'); print(b['name'], b['cv_top1'], b['params']); d=load_dataset('student_test'); print(point_metrics(d, predict_proba(b, d)))"
  ```
- [ ] `final.joblib` < 50 MB; all plots listed above exist in `artifacts/plots/`.
- [ ] `grep -rn evaluator_test src/train.py src/hpo.py src/metrics.py` returns nothing.
- [ ] Scalers live inside `Pipeline`s (fitted per fold, never on the full data before splitting).

## 7. Gotchas

- **Reverse twins**: 92 % of arcs have a geometrically identical reverse arc, so `dist` ties exactly. Only `bearing_diff`/`cos_bearing` (gated by `heading_ok`) can separate them. Nearest-arc Top-1 is 55.6 % for this reason. If a model is stuck near 55 %, it is ignoring heading.
- **Low speed**: heading is meaningless below 1 m/s; `bearing_diff` is set to 90 there. Expect errors concentrated in stop-and-go and pedestrian-speed trips (median speeds range 5.6..41 m/s).
- **Bad-GT trips 33 and 54**: excluded from training and CV. Mention them in the report with the reason (median 122 m / 139 m off route).
- **Unlabelled points** (`label_arc == -1`, >50 m from their aligned arc): not in training, not in metrics. Trip 00000002 has 12 % off-network points.
- **Duplicate (from,to) arcs**: ~22 per trip. Two candidates can have identical features; ties in `p` go to the first row (nearest by sort order). Don't be surprised by a small irreducible error.
- **Long arcs** (up to 9.3 km): `arc_len` is skewed; fine for trees, scaled for LR/MLP. Consider nothing fancier.
- **Huge `dt_prev/dt_next`** (gaps up to 153,064 s in trips 78, 90): heavy tails. Trees don't care; for LR/MLP/kNN consider `np.log1p` inside the pipeline (`FunctionTransformer`) and say so in the report.
- **Worldwide data**: trips span Germany, Moscow, USA, Iraq, Spain... Never add lon/lat/x/y as features; GroupKFold by trip is mandatory because points within a trip are highly correlated (random row splits would leak).
- **Leakage**: no evaluator_test anywhere in your code; student_test is for reporting, not for picking hyperparameters (HPO objective is CV only).
- **Memory**: 1.5M rows x 20 float32 is ~130 MB; fine. RF with 300 trees on full data is the slowest; use `n_jobs=-1` and run it last.
- **trip_id dtype**: zero-padded strings ("00000001"). Reading CSVs without `dtype={"trip_id": str}` turns them into ints and breaks joins with `folds.csv`.

## 8. Who to talk to

- **Merull**: candidate table, features, labels, cache build problems, any PLAN interface change, report integration.
- **B**: needs `final.joblib`, `predict_proba`, `cache/oof_<model>.pkl`, per-model bundles for route continuity, and `point_top1_from_assignment` to score the HMM. Agree on the OOF file format with B before T+2 h.
- **C**: needs `final.joblib` on Laptop 1 and `point_top1_from_assignment` for server-side scoring. Tell C the inference time per trip.

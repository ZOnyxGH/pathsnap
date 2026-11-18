# PathSnap: Technical Plan

Map-matching and route reconstruction. UE24CS352A hackathon.
The plan is decision-first: every number below was measured on the provided data, and every interface in section 6 is fixed. Changing an interface needs the owner's agreement and a commit to this file.

---

## 1. What the data really is (measured)

Full pass over the 80 train trips and 10 student_test trips.

| Fact | Measured value | Consequence |
|---|---|---|
| `.track` columns | `lon lat t_seconds`. Timestamps strictly increasing in all 90 trips | Use as-is. No re-sorting needed |
| `.nodes` columns | `lon lat`, row index = node_id | |
| `.arcs` columns | `from_node to_node`, row index = arc_id. Directed | |
| `.route` | arc_ids in travel order, one per traversed arc (not per point). 59 to 2266 arcs per trip (median 280) | Per-point labels have to be derived (section 3) |
| Route contiguity (`to(r[i]) == from(r[i+1])`) | 99.93 % of pairs overall; 70/80 trips fully contiguous. Worst trip 98.1 % | GT route is a path on the graph. The few breaks are GT gaps, not errors |
| Route arcs revisited | 0.6 % repeat arcs; 0.58 U-turns (`r[i+1]` = reverse of `r[i]`) per trip | Allow U-turns in the HMM, with a penalty |
| Reverse twins | 92 % of arcs have a reverse arc (median 95 %, min 35 %). Duplicate (from,to) pairs: 22 per trip on average | Each road is usually 2 geometrically identical candidates. **Direction must come from heading features**: distance alone cannot separate them |
| **Geography** | **Not Darmstadt only.** The trips are spread worldwide (lon -100..137, lat -35..61): Germany, Moscow, USA, Iraq, Spain, ... Every trip ships its own separate network (all graph hashes differ) | Never use absolute lon/lat as features. Each trip is an independent graph, processed alone |
| Network size per trip | nodes 299..155k (median 9.6k); arcs 649..307k (median 19.5k); extent 3..78 km | Per-trip spatial index. No global graph |
| Track length | 361..19,313 points (median 1,529); duration median 28 min | |
| Sampling | 93.8 % of steps are dt = 1 s. 7.4 gaps > 10 s per trip on average; extreme gaps of 52,810 s and 153,064 s (trips 78, 90) | Gap handling in the HMM is required |
| Speed | median of per-trip medians 12.7 m/s; range 5.6..41 m/s (pedestrian/bike to motorway) | Heading is unreliable at low speed: gate it with a flag |
| Arc length | median 24.5 m; max up to 9.3 km | Long arcs: use segment distance, not distance to the midpoint |
| GPS to nearest GT-route arc | p50 5.6 m, p75 10.8 m, p90 74.9 m (inflated by bad trips, see below). Typical trip: p50 4.5 m, p90 11.8 m, p99 20.4 m | Noise sigma is about 4.5/0.6745 = 6.7 m |
| Bad-GT trips | **00000033** (median 122 m off route, 59 % of points > 50 m) and **00000054** (median 139 m, 87 % > 50 m). Trip 33: route follows the same roads, but ~3,440/5,844 points are dwell clusters >= 50 m off route (track 47.8 km vs route 6.4 km); see docs/EDA_NOTES.md. Trip 00000002 has 12 % off-network points (off-road segment) | Exclude 33 and 54 from training and CV scoring (log it in the report). Points > 50 m from their label are "unlabelled" |
| GT arcs never observed | Only 90 % of route arcs (median 92 %) get a GPS point nearest to them, because short arcs are crossed in under 1 s | Route reconstruction must fill gaps between matched arcs with shortest paths |
| Student test | 10 trips, all have `.route`; clean (p50 4.2..6.8 m, max off50 = 1 %) | Good held-out set for the report table |

**Projection choice.** Per-trip local equirectangular projection centred on the track's mean lat/lon:
`x = (lon-lon0)·cos(lat0)·πR/180`, `y = (lat-lat0)·πR/180`, with R = 6,371,008.8 m.
Reasons: trips span many UTM zones worldwide, pyproj is not installed, and over a 40 km half-extent the scale error is under 1 % (under 0.3 m at the 50 m distances that matter). Everything downstream works in metres.

**Candidate radius / K (measured on points whose label lies within 50 m, 180,762 train points):**

| radius | recall | | K nearest arcs | recall |
|---|---|---|---|---|
| 15 m | 92.9 % | | 1 | 55.9 % (reverse-twin ties) |
| 25 m | 97.8 % | | 2 | 92.2 % |
| 30 m | 98.6 % | | 5 | 98.1 % |
| 40 m | 99.5 % | | 8 | 99.3 % |
| 50 m | 100 % | | 16 | 99.6 % |
| | | | 32 | 99.98 % |

Arcs within 50 m: median 6 per point, p95 28.

**Naive baselines (Top-1 on labelled points):** nearest arc 55.1 % train / 53.7 % student_test; nearest + heading rule 83.1 % / 84.2 % (after heading-aware labels). Every model has to beat ~84 %.

---

## 2. Decisions at a glance

| Item | Decision |
|---|---|
| Labels | Monotonic DP alignment of points to route positions (section 3) |
| Candidates | All arcs with segment distance ≤ **50 m**, keep the **16 nearest**. If none lie within 50 m, take the 4 nearest within 200 m (`fallback=1`). If still none, the point has no candidates (HMM skips it) |
| Excluded trips | 00000033 and 00000054 (bad GT) from training/CV. Reported separately |
| CV | `GroupKFold(n_splits=5)` on trip_id, over the 78 clean train trips. Folds are deterministic (GroupKFold over trips sorted by trip_id), saved to `artifacts/results/folds.csv` and reused by every model, HPO and the HMM evaluation |
| Final model (expected) | Gradient boosting (LightGBM or XGBoost, chosen by CV evidence), tuned with Optuna TPE |
| Decoder | HMM + Viterbi over candidate arcs, emission = classifier log-probability |
| Primary metric | Point Top-1 accuracy on labelled points. Also Macro F1, log loss, miss rate, route F1, continuity |
| Storage | Candidate tables cached as pandas pickle (`cache/`, gitignored, rebuilt locally). pyarrow is not installed, so no parquet |
| New deps | `lightgbm`, `optuna`, `folium` (py3.14 wheels confirmed) |

---

## 3. Per-point labels (the key design problem)

The `.route` is an ordered arc list, with about 0.15 arcs per GPS point. Each point needs "the arc I was on".

**Method: monotonic alignment by dynamic programming.**
- Cost matrix `C[i, j]` = projected segment distance from point i to route arc `route[j]` (n × m; the largest trip is 19k × 2.3k, about 44M floats, fine).
- Robust cost `c = min(C, 50)`: a single off-road point can't drag the alignment.
- Heading term: for points moving at >= 1 m/s, `c += 2*(1 - cos(heading - arc bearing))`, which separates reverse twins at U-turns (equal distance).
- Find a non-decreasing `j(i)` that minimises `Σ c[i, j(i)]`. Recurrence: `D[i, j] = c[i, j] + min_{j' ≤ j} D[i-1, j']`, a prefix-min per row, O(n·m), under 1 s per trip in numpy. Backtrack with stored argmins.
- `label_arc[i] = route[j(i)]`. A point is **labelled** only if `C[i, j(i)] ≤ 50 m`; otherwise `label_arc = -1` (off-network / GT mismatch). Unlabelled points are dropped from training and from point metrics, but still go through the HMM.

**Why not the nearest route arc:** when the route uses a road in both directions, or loops, the nearest route arc is wrong in time. The measured agreement between DP and nearest-route-arc is median 97 %, but only **54 %** on trip 11 and 74 % on trip 6 (out-and-back roads where nearest picks the twin). The monotonic constraint resolves the twin and loop ambiguity because travel order is known. Measured `j(i)` jumps over GT arcs that were never observed. That is expected, and is why `route_cov` is about 90 %.

Candidate-row label: `label = 1` iff `arc_id == label_arc[pt_idx]`. If the labelled arc isn't among the candidates, the point counts as a **candidate miss** (target under 0.5 %).

---

## 4. Features (one row = one (point, candidate arc) pair)

Heading `h_i` = bearing of `xy[i+1] - xy[i-1]` (centred difference; one-sided at the ends). `heading_ok = speed ≥ 1.0 m/s`.

| Column | Definition |
|---|---|
| `dist` | point-to-segment distance (m) |
| `dist_rank` | rank of `dist` among the point's candidates (0 = nearest) |
| `dist_gap` | `dist - min dist of point` |
| `dist_ratio` | `dist / (min dist of point + 1)` |
| `proj_frac` | projection parameter t ∈ [0,1] on the arc |
| `edge_proj` | `min(proj_frac, 1-proj_frac) * arc_len` (m from the nearest arc end; values near 0 mean ambiguous at junctions) |
| `arc_len` | arc length (m) |
| `bearing_diff` | `|h_i - arc_bearing|`, wrapped to [0, 180] deg; set to 90 if `heading_ok = 0` |
| `cos_bearing` | `cos(bearing_diff)` (signed; separates reverse twins) |
| `heading_ok` | 0/1 |
| `speed` | `|xy[i+1]-xy[i-1]| / (t[i+1]-t[i-1])` (m/s) |
| `dt_prev`, `dt_next` | seconds to the neighbour points (gap awareness) |
| `n_cand` | number of candidates of the point |
| `has_twin` | reverse arc also among the candidates |
| `out_deg`, `in_deg` | out-degree of `to_node`, in-degree of `from_node` |
| `prev_link` | 1 if the arc equals, or shares a node with (`from == to(prev)` or same arc), any of the previous point's 3 nearest candidates. Cheap 1-hop connectivity |
| `next_link` | same, with the next point's 3 nearest |
| `fallback` | 1 if it came from the 200 m fallback |

Forbidden: absolute lon/lat/x/y, trip_id, timestamps as absolute values (no generalisation, risk of leakage).

Size estimate: ~200k points × ~7 candidates ≈ 1.4–1.8M rows. Train full-data for tree models; subsample 300k rows (grouped by trip) for SVM / kNN / MLP during model selection.

---

## 5. Models, selection, HPO

**Six families** (all sklearn-compatible, all trained on the same features with the same folds):

| # | Model | Family | Notes |
|---|---|---|---|
| 1 | LogisticRegression (+StandardScaler) | linear | `class_weight="balanced"` |
| 2 | GaussianNB | probabilistic / generative | baseline |
| 3 | RandomForestClassifier | bagging ensemble | n_estimators=300, min_samples_leaf=20 |
| 4 | XGBClassifier | boosting | hist |
| 5 | LGBMClassifier | boosting (2nd impl.) | |
| 6 | MLPClassifier (+scaler) | neural | (64, 32), early_stopping |
| (7) | KNeighborsClassifier | instance-based | optional, subsampled |

Selection: the best mean **CV point Top-1** across 5 GroupKFold folds wins; log loss breaks ties (the HMM consumes probabilities, so calibration matters). Record the train time and the inference time per 1k rows.

**HPO:** Optuna `TPESampler(seed=42)`, 40 trials (cut to 25 if time is short), objective = mean CV Top-1 (GroupKFold, 5 folds, groups = trip). Pruner: `MedianPruner` after fold 2. Trials logged to `artifacts/hpo/trials.csv`. Search space if the winner is LightGBM:
- `n_estimators` 200–1500, `learning_rate` 0.01–0.3 (log), `num_leaves` 15–255 (log), `max_depth` -1 or 4–12, `min_child_samples` 10–200 (log), `subsample` 0.5–1 (`subsample_freq=1`), `colsample_bytree` 0.5–1, `reg_lambda` 1e-3–10 (log), `reg_alpha` 1e-3–10 (log).
- For XGBoost, the same with `max_depth` 3–10 and `min_child_weight` 1–20 instead of leaves/min_child.

Refit the best params on all 78 clean train trips, then save `artifacts/models/final.joblib`.

**Metric definitions (shared, in `src/metrics.py`):**
- *Top-1*: per labelled point, the argmax-p candidate equals `label_arc`. A miss counts as wrong.
- *Macro F1*: candidate-level binary prediction `pred = 1` iff the candidate is the point's argmax; macro over classes {0, 1}.
- *Log loss*: candidate-level binary log loss of `p` vs `label`.
- *Candidate miss rate*: labelled points whose `label_arc` is not among the candidates.
- Route metrics: see section 7.

---

## 6. Repo layout and FIXED interfaces

```
codebase/
  README.md              # C (run instructions), lead integrates results
  requirements.txt       # numpy pandas scipy scikit-learn xgboost lightgbm optuna folium fastapi uvicorn matplotlib seaborn joblib networkx
  .gitignore             # cache/, __pycache__/, *.pkl, /data
  docs/PLAN.md
  src/
    __init__.py
    data.py              # Merull: paths, loading, projection, labels
    features.py          # Merull: candidates + features + dataset build (CLI)
    metrics.py           # A: point metrics
    train.py             # A: all models, CV, plots, confusion matrices (CLI)
    hpo.py               # A: Optuna (CLI)
    hmm.py               # B: transitions, Viterbi, route reconstruction, route metrics
    predict.py           # B: end-to-end single-trip inference (used by eval + API)
    evaluate.py          # B: classifier-only vs HMM on CV folds + student_test (CLI)
    maps.py              # B: folium maps
  api/server.py          # C
  ui/index.html          # C (static page, Laptop 2)
  artifacts/             # committed
    sample/              # Merull: small candidate CSV for teammates to start on
    models/final.joblib
    results/*.csv
    plots/*.png
    hpo/trials.csv
    maps/*.html
  cache/                 # gitignored: {split}_candidates.pkl
```

Raw data is not committed. `src/data.py` resolves `DATA_ROOT = os.environ.get("PATHSNAP_DATA", <codebase>/..)`, which is the Hackathon folder containing `train/`, `student_test/`, and later `evaluator_test/`. Run everything as modules from `codebase/`: `python -m src.features train`.

### 6.1 `src/data.py` (Merull)

```python
DATA_ROOT: Path
EXCLUDED_TRIPS = {"00000033", "00000054"}

class Trip(NamedTuple):
    trip_id: str
    t: np.ndarray          # (n,) float seconds
    lonlat: np.ndarray     # (n,2) GPS lon,lat
    xy: np.ndarray         # (n,2) metres, local projection
    origin: tuple          # (lat0, lon0)
    nodes_lonlat: np.ndarray  # (N,2)
    nodes_xy: np.ndarray      # (N,2)
    arcs: np.ndarray       # (M,2) int64 from,to
    arc_len: np.ndarray    # (M,) metres
    route: np.ndarray | None  # (R,) int64 arc ids, None if not loaded

def list_trips(split: str) -> list[str]
def load_trip(split: str, trip_id: str, with_route: bool = True) -> Trip
def project(lonlat: np.ndarray, origin: tuple) -> np.ndarray
def align_labels(trip: Trip, max_dist: float = 50.0) -> tuple[np.ndarray, np.ndarray]
    # -> label_arc (n,) int64 (-1 = unlabelled), label_dist (n,) float
```

### 6.2 `src/features.py` (Merull)

```python
RADIUS, K, FALLBACK_RADIUS, FALLBACK_K = 50.0, 16, 200.0, 4
FEATURES: list[str]   # exactly the section-4 columns, in this order

def build_candidates(trip: Trip) -> pd.DataFrame       # no labels, usable at inference
def add_labels(cands: pd.DataFrame, trip: Trip) -> pd.DataFrame
def build_dataset(split: str) -> pd.DataFrame          # all trips, with labels, cached to cache/{split}_candidates.pkl
def load_dataset(split: str) -> pd.DataFrame           # read the cache (build if missing)
# CLI: python -m src.features train|student_test   (also writes artifacts/sample/ for 3 trips)
```

**Candidate table contract** (one row per (point, candidate)):

| column | dtype | note |
|---|---|---|
| `trip_id` | str | |
| `pt_idx` | int32 | row index in `.track` |
| `arc_id` | int64 | |
| *FEATURES…* | float32 | section 4 |
| `label` | int8 | 1/0; only present after `add_labels` |
| `label_arc` | int64 | point's true arc, -1 if unlabelled (repeated on each row) |

Rows are sorted by `(trip_id, pt_idx, dist)`. Points with no candidates have no rows.

### 6.3 Model bundle (A writes, B/C read)

```python
# artifacts/models/final.joblib
{"name": "lgbm", "model": fitted_estimator, "features": FEATURES, "params": {...}, "cv_top1": float}

def predict_proba(bundle: dict, cands: pd.DataFrame) -> np.ndarray   # in src/train.py, P(label=1) per row
```

### 6.4 `src/metrics.py` (A)

```python
def point_metrics(cands: pd.DataFrame, p: np.ndarray) -> dict
    # keys: top1, macro_f1, log_loss, miss_rate, n_points
def point_top1_from_assignment(label_arc: np.ndarray, pred_arc: np.ndarray) -> float
    # used to score HMM output with the same definition (ignores label_arc == -1)
```

### 6.5 `src/hmm.py`, `src/predict.py` (B)

```python
def viterbi(trip: Trip, cands: pd.DataFrame, p: np.ndarray, beta=..., lam=1.0, sigma=6.7) -> tuple[np.ndarray, list[int]]
    # -> pred_arc (n,) int64 (-1 = no candidates), break indices
def reconstruct_route(trip: Trip, pred_arc: np.ndarray) -> list[int]   # full arc sequence with gap filling
def route_metrics(trip: Trip, pred_route: list[int], ref_route: np.ndarray) -> dict
    # keys: route_precision, route_recall, route_f1 (length-weighted), continuity, n_breaks

def predict_trip(split: str, trip_id: str, bundle: dict, use_hmm: bool = True) -> dict
    # -> {"trip_id", "pred_arc": list[int], "route": list[int], "route_lonlat": list[[lon,lat]...], "timing_ms": {...}}
    # NEVER reads .route (load_trip(..., with_route=False))
```

### 6.6 API contract (C)

- `GET /trips` → `{"split": "evaluator_test", "trips": ["0000xxxx", ...]}`
- `POST /predict` body `{"trip_id": "..."}` → `{"trip_id", "route_lonlat": [[lon,lat],...], "gps_lonlat": [[lon,lat],...], "metrics": {"top1", "route_f1", "route_precision", "route_recall", "continuity"}, "inference_ms"}`
- `GET /map/{trip_id}` → folium HTML of GPS + predicted route (no reference layer)
- Ground truth is loaded **only inside** the scoring function after prediction, and is never serialised to the response except as scalar metrics.
- Each request is logged to the terminal: `[ts] client_ip  METHOD path  trip_id  status  latency_ms`.

---

## 7. HMM / Viterbi specification (B)

- **Hidden states at point i**: its candidate arcs (≤ 16). Points with no candidates are skipped; the transition is computed across the skip.
- **Emission**: `log e = lam·log(clip(p, 1e-6, 1)) + (1-lam)·(-0.5·(dist/sigma)²)`, sigma = 6.7 m (measured). Default `lam = 1`. Ablation: lam ∈ {0 (classic Newson–Krumm, no ML), 0.5, 1}.
- **Transition** between consecutive kept points u→v with candidates a→b:
  - `gc = |xy[v] - xy[u]|`
  - route distance `rd`: if `a == b`: `(t_b - t_a)·len_a` (allow small negative t_b < t_a up to 10 m of jitter, using |·|). Else `(1-t_a)·len_a + SP(to(a), from(b)) + t_b·len_b`, where SP is the directed shortest path on arc lengths.
  - `log T = -|rd - gc| / beta`, beta tuned over {2, 5, 10, 20} m on CV (start at 5).
  - U-turn (`b` = reverse of `a`): extra penalty of -3 (log space).
  - SP via `scipy.sparse.csgraph.dijkstra(G, indices=unique to-nodes of step u, limit=L)`, with `L = max(500, 3·gc + 200, v_max·dt)` and v_max = 50 m/s. Cache per source node within a trip. Unreachable within L means log T = -inf.
- **Breaks**: if every transition into step v is -inf (disconnected, or out of range), close the current Viterbi segment and restart at v with emissions only. Record the break.
- **Gaps**: `dt > 10 s` raises L through `v_max·dt` and scales beta by `dt/1`. `dt > 120 s` forces a break.
- **Route reconstruction**: Viterbi arcs, collapse consecutive duplicates, and insert SP arcs between consecutive distinct arcs that aren't adjacent (predecessors from the same Dijkstra). Across breaks, concatenate (continuity drops; that's reported).
- **Route metrics**: length-weighted precision/recall/F1 over arc-id sets (pred vs `.route`); continuity = fraction of consecutive predicted pairs with `to(a) == from(b)`; n_breaks.
- Speed budget: under 5 s per typical trip on a laptop. If it's too slow, drop points within 5 m of the previous kept point (standard Newson–Krumm thinning) and copy their assignment from the neighbours.

---

## 8. Leakage rules

- Evaluator trips are never loaded by `train.py`, `hpo.py`, or `features.build_dataset`. Only `api/server.py` and `evaluate.py --split evaluator_test` read them.
- `predict_trip` loads with `with_route=False`. The ground truth is read separately for scoring only.
- Feature scaling lives inside sklearn `Pipeline`s, fitted per CV fold.
- sigma and beta are estimated from train trips only.
- No hard-coded trip IDs except `EXCLUDED_TRIPS` (data-quality, justified above).

---

## 9. Work split and dependencies

| Person | Owns | Can start immediately on | Blocked by |
|---|---|---|---|
| **Merull** (lead) | `data.py`, `features.py`, `artifacts/sample/`, EDA plots, report integration | everything (critical path) | none |
| **A** | `metrics.py`, `train.py`, `hpo.py`, model plots/confusion matrices, `cv_results.csv` | `train.py` + `metrics.py` against a synthetic DataFrame with the section 6.2 columns; install lightgbm/optuna | sample CSV (T+45 min), full cache (T+1.5 h) |
| **B** | `hmm.py`, `predict.py`, `evaluate.py`, `maps.py` | Dijkstra/transition code and the folium map using `load_trip` (T+20 min); Viterbi with lam = 0 (needs only `dist`, `proj_frac`) | `build_candidates` (T+1 h); `final.joblib` for lam = 1 |
| **C** | `api/server.py`, `ui/index.html`, two-laptop networking, `README.md` | FastAPI skeleton with a stub `predict_trip` returning the GPS track as the "route"; UI; firewall/IP setup; request logging | `predict_trip` (B) for the real model |

**Checkpoints (T = start):**
1. **T+20 min**: `data.py` pushed (`load_trip`, `project`). Interfaces in this file are frozen.
2. **T+45 min**: `align_labels`, plus `artifacts/sample/train_sample_candidates.csv` (3 trips).
3. **T+1 h**: `build_candidates` done. B runs the lam = 0 HMM end to end. C wires the real `predict_trip`. **The first end-to-end demo (API → UI) must work here**, with the heading-rule baseline or the Gaussian HMM.
4. **T+1.5 h**: full `cache/train_candidates.pkl` + `student_test`. A runs the 6-model CV.
5. **T+2.5 h**: winner + HPO running; B swaps in `final.joblib` (lam = 1) and tunes beta.
6. **T+3 h**: results CSVs and maps committed. Two-laptop dry run with the student_test trips standing in for evaluator trips.

**Git:** each person edits only their own files. Branch per person, merged to `main` by Merull. Small artifacts (CSVs, PNGs, model ≤ 50 MB) are committed; `cache/` never is.

---

## 10. Final report table (target columns)

`model | CV Top-1 | CV Macro F1 | CV log loss | miss rate | route F1 | continuity | student-test Top-1 | student-test Macro F1 | train s | infer ms/trip`

Rows: 6 classifiers (classifier-only argmax), the tuned winner, the winner + HMM, and the Gaussian HMM (lam = 0) baseline, plus the two naive baselines (nearest 55.1 %, nearest + heading 83.1 %).

# Teammate B handoff: HMM/Viterbi, route reconstruction, evaluation, folium maps

Read `docs/PLAN.md` sections 1, 3, 6.1-6.5, 7, 8, 10 once. This doc is the condensed version for your part.

## 1. Project in 5 lines

- Input per trip: a GPS track (`lon lat t_seconds`, ~1 Hz) plus that trip's own directed road network. Train/student_test trips also have `.route` (ground-truth arc sequence, in travel order).
- Merull generates up to 16 candidate arcs per GPS point (within 50 m) with features; A trains a classifier giving P(candidate is the true arc).
- **Your job:** Viterbi over the candidates (emission = classifier log-prob, transition = road-network shortest-path consistency), turn the per-point arcs into a full connected route, score it, and draw it with folium.
- `predict_trip` is the single inference entry point used by `evaluate.py` and by C's API in the live demo.
- The report must show classifier-only vs classifier+HMM, a lam ablation (Gaussian HMM baseline, no ML), beta tuning, and folium maps.

## 2. Your deliverables

### `src/hmm.py` (PLAN 6.5, fixed signatures)

```python
def viterbi(trip: Trip, cands: pd.DataFrame, p: np.ndarray, beta=5.0, lam=1.0, sigma=6.7) -> tuple[np.ndarray, list[int]]
    # -> pred_arc (n,) int64 (-1 = no candidates), break indices
def reconstruct_route(trip: Trip, pred_arc: np.ndarray) -> list[int]   # full arc sequence with gap filling
def route_metrics(trip: Trip, pred_route: list[int], ref_route: np.ndarray) -> dict
    # keys: route_precision, route_recall, route_f1 (length-weighted), continuity, n_breaks
```

(PLAN writes `beta=...`; the agreed default is 5.0 until tuning says otherwise. Once tuned, change the default to the tuned value so the API picks it up.)

### `src/predict.py` (PLAN 6.5, fixed)

```python
def predict_trip(split: str, trip_id: str, bundle: dict, use_hmm: bool = True) -> dict
    # -> {"trip_id", "pred_arc": list[int], "route": list[int], "route_lonlat": list[[lon,lat]...], "timing_ms": {...}}
    # NEVER reads .route (load_trip(..., with_route=False))
```

Pipeline: `load_trip(split, trip_id, with_route=False)` -> `build_candidates(trip)` -> `p = predict_proba(bundle, cands)` -> `viterbi(...)` (or per-point argmax if `use_hmm=False`) -> `reconstruct_route` -> `route_lonlat`.
- `route_lonlat`: `nodes_lonlat[arcs[route[0], 0]]` followed by `nodes_lonlat[arcs[a, 1]]` for each arc `a` in `route`. Plain Python floats (`.tolist()`), JSON-serialisable.
- `timing_ms`: `{"load", "candidates", "classifier", "viterbi", "reconstruct", "total"}`.
- Allow `bundle=None` -> lam=0 path (p = ones). This is what runs at T+1 h before `final.joblib` exists, and it is the Gaussian HMM baseline.
- Keep it stateless (no globals carried between calls). C loads the bundle once at server startup and passes it in on every request.

### `src/evaluate.py` (CLI)

```
python -m src.evaluate --split train          # CV: uses A's out-of-fold probabilities, folds.csv
python -m src.evaluate --split student_test   # uses final.joblib
python -m src.evaluate --split train --tune-beta          # beta in {2,5,10,20}
python -m src.evaluate --split train --lam-ablation       # lam in {0, 0.5, 1}
python -m src.evaluate --split student_test --bundle cache/models/rf.joblib --no-hmm   # route continuity per classifier for A's table
python -m src.evaluate --split evaluator_test             # ONLY at the very end, after everything is frozen
```

Modes compared per trip: `classifier` (argmax p, then `reconstruct_route`), `hmm` (lam=1, tuned beta), `gauss_hmm` (lam=0), plus `lam=0.5` in the ablation.

Outputs (committed):

| File | Content |
|---|---|
| `artifacts/results/hmm_comparison.csv` | per (split, trip_id, mode): `top1, route_precision, route_recall, route_f1, continuity, n_breaks, ms_total` |
| `artifacts/results/hmm_summary.csv` | mean per (split, mode); this feeds the report table rows "winner + HMM" and "Gaussian HMM" |
| `artifacts/results/hmm_beta.csv` | beta x {top1, route_f1, continuity} on train CV (and student_test) |
| `artifacts/results/hmm_lam.csv` | lam x metrics |
| `artifacts/results/classifier_route_metrics.csv` | per classifier (from A's bundles, `--no-hmm`): route F1, continuity |
| `artifacts/plots/hmm_vs_classifier.png` | grouped bars: Top-1, route F1, continuity, classifier-only vs HMM |
| `artifacts/plots/hmm_per_trip_delta.png` | per-trip Top-1 (HMM minus classifier), sorted; shows where HMM hurt |
| `artifacts/plots/hmm_beta.png`, `artifacts/plots/hmm_lam.png` | sweeps |

### `src/maps.py`

Proposed signature (not in PLAN yet; agree with C, then ask Merull to add it to 6.5):

```python
def trip_map(trip: Trip, pred_route: list[int] | None = None, ref_route: np.ndarray | None = None,
             show_network: bool = True, out_html: str | None = None) -> folium.Map
```

Required layers (brief section 6): each a `folium.FeatureGroup` so `folium.LayerControl()` can toggle it.
- **GPS observations**: blue `CircleMarker`s (radius 2). For long tracks (up to 19k points) draw a thin blue `PolyLine` plus every ~5th point as a marker, or the HTML gets huge.
- **Road network**: thin grey lines, only arcs near the track (both nodes inside the track bounding box + 300 m, or arcs from `build_candidates`). Never draw all 307k arcs.
- **Reference route** (green, weight 5, opacity 0.6) when `ref_route` is given. Never in the API's map.
- **Predicted route** (red, weight 3) on top.
- Start/end markers (green/red icons), legend (a small fixed HTML box added with `m.get_root().html.add_child(folium.Element(...))`), `m.fit_bounds(...)` on the track extent.

C serves `trip_map(trip_without_route, pred_route=result["route"]).get_root().render()` at `GET /map/{trip_id}`: no reference layer there.

Committed maps in `artifacts/maps/`: at least 3 student_test trips (one good, one where HMM helped most, one where it failed), one train trip with a huge gap (78 or 90), each with GPS + network + reference + classifier-only + HMM layers. One screenshot goes in the report.

## 3. Inputs you consume

**`Trip`** (`src/data.py`, available from T+20 min):

```python
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

load_trip(split, trip_id, with_route=True) -> Trip
align_labels(trip) -> (label_arc (n,) int64 with -1 = unlabelled, label_dist (n,))
seg_dist(P, A, B) -> (dist, t)      # helper already in data.py, reuse it
EXCLUDED_TRIPS = {"00000033", "00000054"}
```

**Candidates** (`src/features.py`, `build_candidates(trip)` from T+1 h; no labels, safe at inference): one row per (point, candidate), sorted by `(trip_id, pt_idx, dist)`. You need `pt_idx` (int32), `arc_id` (int64), `dist` (m, point-to-segment), `proj_frac` (t in [0,1] along the arc). Points with no candidates have no rows. Before T+1 h, use `artifacts/sample/train_sample_candidates.csv` (3 train trips, same columns + labels; read with `dtype={"trip_id": str}`) together with `load_trip("train", <that trip_id>)`.

**Probabilities**:
- `from src.train import predict_proba` + `artifacts/models/final.joblib` (bundle `{"name","model","features","params","cv_top1"}`), untuned version ~T+2.5 h, tuned ~T+3 h.
- Before that: lam=0, which needs only `dist` and `proj_frac`. Pass `p = np.ones(len(cands))`.
- For CV evaluation on train you must **not** use `final.joblib` (it was fitted on all train trips: leakage). Use A's out-of-fold file `cache/oof_<winner>.pkl` (`trip_id, pt_idx, arc_id, p`), merged onto `build_candidates(trip)` rows on `(pt_idx, arc_id)`. `artifacts/results/folds.csv` (`trip_id, fold`) gives per-fold averages.

**Scoring** (`src/metrics.py`, A): `point_top1_from_assignment(label_arc, pred_arc)`. Ground truth for scoring: `trip_gt = load_trip(split, tid)` (with route), `label_arc, _ = align_labels(trip_gt)`, `route_metrics(trip_gt, route, trip_gt.route)`. Load GT only after prediction.

## 4. Step-by-step tasks

| By | Task |
|---|---|
| T+20 min | Graph + shortest paths on a real train trip via `load_trip`: build `csr_matrix`, `dijkstra(..., limit=L, return_predecessors=True)`, per-source cache. `trip_map` with GPS + network + reference layers (no prediction yet) |
| T+45 min | Transition matrix for one step (K_u x K_v) vectorised; reverse-arc lookup for the U-turn penalty; `route_metrics` |
| T+1 h | `viterbi` lam=0 end to end on 1 trip, `reconstruct_route`, `predict_trip(bundle=None)`. Push so C can wire the real pipeline. **First API->UI demo depends on this** |
| T+1.5 h | Speed: < 5 s per typical trip. Run lam=0 on all student_test trips, first `hmm_comparison.csv` rows |
| T+2.5 h | Swap in `final.joblib` (lam=1). Train-CV evaluation with OOF probabilities. Beta sweep {2,5,10,20}, lam ablation {0,0.5,1}. Set tuned beta as default |
| T+3 h | All CSVs, plots, maps committed. Classifier-only route continuity for A's 6 models. Push, PR. Support the two-laptop dry run |
| after | Draft report sections 5 and 6 with real numbers; pick failure cases and explain them |

### Implementation notes (PLAN 7, condensed)

- **States** at point i: its candidate arcs (<= 16). Points with no candidates are skipped; transition goes from the last kept point to the next one.
- **Emission**: `log e = lam*log(clip(p, 1e-6, 1)) + (1-lam)*(-0.5*(dist/sigma)**2)`, sigma = 6.7 m.
- **Transition** u->v, candidates a->b: `gc = |xy[v]-xy[u]|`.
  - `a == b`: `rd = |t_b - t_a| * len_a` (backward moves up to ~10 m are GPS jitter).
  - else: `rd = (1-t_a)*len_a + SP(to(a), from(b)) + t_b*len_b`.
  - `log T = -|rd - gc| / beta`; U-turn (b is the reverse of a): extra -3.
  - `SP` from `scipy.sparse.csgraph.dijkstra(G, indices=unique to-nodes of step u, limit=L)`, `L = max(500, 3*gc + 200, 50*dt)`. Cache per source node. `inf` -> log T = -inf.
- **Breaks**: if every transition into v is -inf, close the segment (backtrack it), restart at v with emissions only, record v in `breaks`.
- **Gaps**: `dt > 10 s`: L grows via `50*dt`, beta is scaled by `dt`. `dt > 120 s`: forced break.
- **Graph build**: `csr_matrix((len, (from, to)), shape=(N, N))` **sums duplicate (from,to) entries** (there are ~22 per trip). Deduplicate first, keeping the shortest arc per pair, and keep a `(from,to) -> arc_id` dict for path recovery. Add 1e-3 m to lengths so zero-length arcs remain edges.
- **Route reconstruction**: collapse consecutive duplicates; for consecutive distinct arcs a, b with `to(a) != from(b)`, insert the SP arcs from `to(a)` to `from(b)` using predecessors (rerun Dijkstra from that node if not cached). If unreachable, just concatenate (continuity drops; that is reported).
- **Route metrics**: arc-id sets (direction matters, twins are different arcs). `P = len(pred ∩ ref)/len(pred)`, `R = len(pred ∩ ref)/len(ref)` with lengths = `sum(arc_len[...])`; F1 harmonic. `continuity` = fraction of consecutive predicted pairs with `to(a) == from(b)`; `n_breaks` = number of pairs where it fails.
- **Speed**: if a trip takes > 5 s, thin points within 5 m of the previous kept point and copy the neighbour's assignment. Log `timing_ms`; C shows it in the demo.
- Leave a small self-check in `hmm.py` (`python -m src.hmm`): a 3-arc toy graph where the HMM must pick the connected path over an emission-favoured disconnected arc.

## 5. Report sections you own

### Section 5: HMM and Viterbi route reconstruction (answer all 6)

1. What are the hidden states in your implementation?
2. How are emission probabilities calculated?
3. How are transition probabilities calculated?
4. How are missing candidates or large GPS gaps handled?
5. What happens when consecutive GPS observations suggest disconnected road segments?
6. Did HMM decoding improve the final route? Support with experimental results.

Required comparison: classifier-only predictions vs classifier + HMM/Viterbi (Top-1, route F1, continuity, per trip and mean). Discuss improvements **and cases where HMM did not help** (use `hmm_per_trip_delta.png` + a map of the worst trip). Also: Viterbi recurrence written out, the lam ablation (lam=0 is the classic Newson-Krumm HMM with no ML), beta sweep, and the statement that decoder parameters (beta, sigma, lam) were tuned on train CV / student_test only.

### Section 6: Geographical visualisation using folium

Minimum layers: original GPS, relevant road-network geometry, reference route (where available), predicted route. Maps must:
1. Use distinguishable colours and markers for different route layers.
2. Include clear labels or a legend.
3. Have an appropriate extent and zoom level.
4. Make it easy to compare the observed GPS trajectory with the predicted route.
5. Include layer controls.

Submission: at least one map screenshot in the PDF, the HTML maps and the generating code committed. Provide Merull a map of one train trip (GPS + network + reference) for the EDA section too (brief 2.2 asks for at least one folium map there).

Also supply A the route F1 / continuity numbers for each classifier (PLAN 10 table and brief 3.1 "Route continuity" column).

## 6. Definition of done + verification

- [ ] `python -m src.hmm` self-check passes.
- [ ] `python -c "from src.predict import predict_trip; r = predict_trip('student_test', '00000007', None); print(len(r['route']), r['timing_ms'])"` works (lam=0, no model).
- [ ] With the model: `python -c "import joblib; from src.predict import predict_trip; b = joblib.load('artifacts/models/final.joblib'); print(predict_trip('student_test', '00000007', b)['timing_ms'])"`; total < 5000 ms for typical trips.
- [ ] `predict_trip` contains no `with_route=True` and never touches `.route` (grep it).
- [ ] `python -m src.evaluate --split student_test` writes `hmm_comparison.csv` with classifier / hmm / gauss_hmm rows for all 10 trips; HMM continuity close to 1.0 and Top-1 >= classifier-only on average (if not, investigate before reporting).
- [ ] Train CV evaluation uses OOF probabilities, never `final.joblib`.
- [ ] Maps open in a browser, layer control toggles every layer, legend visible, zoomed to the track.
- [ ] All CSVs/plots/maps from section 2 committed; each HTML map is a few MB at most.
- [ ] Every `src/` module runs on all 10 student_test trips without exceptions (C's demo must work for every trip, not one).

## 7. Gotchas

- **Reverse twins**: 92 % of arcs have a reverse arc with identical geometry. Emission alone can't separate them at low speed; transitions help (driving the twin needs a U-turn, -3 penalty). Route metrics treat twins as different arcs, as they should.
- **Bad-GT trips 33 and 54**: large fractions of points > 50 m from route (dwell clusters / GT mismatch; median 122 m / 139 m off). Exclude from beta tuning and CV averages; mention separately. Trip 00000002 has 12 % off-network points.
- **Long arcs** (up to 9.3 km): always use `proj_frac * arc_len` positions, never arc midpoints. A point can sit on the same long arc for minutes (a == b case).
- **Huge time gaps**: 7.4 gaps > 10 s per trip on average; trips 78 and 90 have 52,810 s and 153,064 s gaps. These must force a break (dt > 120 s), otherwise L = 50*dt makes Dijkstra explore the whole graph.
- **Big networks**: up to 155k nodes / 307k arcs per trip. Build the graph once per trip; always pass `limit=L` to Dijkstra; only run it from unique source nodes.
- **GT gaps**: only ~90 % of route arcs get a GPS point nearest to them (short arcs crossed in < 1 s), so gap filling with shortest paths is required for good route recall. 4/80 train trips have small discontinuities in GT itself; continuity of the reference is not 100 % either.
- **Worldwide coordinates**: trips are in Germany, Moscow, USA, Iraq, Spain... Never hardcode a map centre; always `fit_bounds` on the track. Everything numeric happens in the per-trip local `xy` metres.
- **Unlabelled points** (`label_arc == -1`) still go through Viterbi but are ignored by `point_top1_from_assignment`.
- **Leakage**: `predict_trip` loads with `with_route=False`; beta/lam/sigma tuned on train + student_test only; `evaluate.py --split evaluator_test` runs once at the end and changes nothing.
- **Determinism**: ties in Viterbi (twins with equal scores) should resolve the same way every run (use `argmax`, which takes the first).

## 8. Who to talk to

- **Merull**: `load_trip`, `build_candidates`, labels, the candidate miss rate, adding `trip_map` to PLAN, getting your branch merged.
- **A**: `final.joblib`, `predict_proba`, OOF probability file format (`cache/oof_<model>.pkl`), per-model bundles for classifier-only route metrics, `point_top1_from_assignment`.
- **C**: consumes `predict_trip` and `trip_map`. Tell C immediately if the return dict or timing changes. C will report any trip that crashes in the API; fix those first.

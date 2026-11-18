# PathSnap

Map matching and route reconstruction for noisy GPS tracks (UE24CS352A hackathon). Each GPS point is
matched to a road arc with a gradient-boosting classifier over candidate arcs, then an HMM/Viterbi
decoder turns the per-point probabilities into a connected route. The full design is in `docs/PLAN.md`.

## Setup

```
pip install -r requirements.txt
```

Python 3.10+ (developed on 3.14). Run everything as modules from `codebase/`.

## Data

Raw data is not in the repo. Expected layout, one folder above `codebase/`:

```
Hackathon/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
  student_test/<trip_id>/...
  evaluator_test/<trip_id>/...      (later)
  codebase/
```

To use another location set `PATHSNAP_DATA` to the folder that contains `train/` and `student_test/`.

## Build the candidate tables

```
python -m src.features train
python -m src.features student_test
```

Writes `cache/<split>_candidates.pkl` (gitignored, about 1 minute for train) and a 3-trip CSV to
`artifacts/sample/`. Prints trip/point/row counts, candidate miss rate and the two naive baselines.
In code: `from src.features import load_dataset, FEATURES`.

## Layout

```
src/data.py       loading, projection, label alignment
src/features.py   candidates, features, dataset cache
src/metrics.py    point metrics
src/train.py      models, CV, final bundle
src/hpo.py        Optuna search
src/hmm.py        Viterbi, route reconstruction, route metrics
src/predict.py    end-to-end single-trip inference
src/evaluate.py   classifier vs HMM evaluation
src/maps.py       folium maps
api/server.py     FastAPI server
ui/index.html     web UI
tests/            python -m pytest
artifacts/        committed results, models, plots, samples
cache/            local candidate tables (not committed)
```

## Team

| Module | Owner |
|---|---|
| `data.py`, `features.py`, `artifacts/sample/` | Merull (lead) |
| `metrics.py`, `train.py`, `hpo.py` | A |
| `hmm.py`, `predict.py`, `evaluate.py`, `maps.py` | B |
| `api/server.py`, `ui/index.html`, `README.md` | C |

## Deployment (two laptops)

Laptop 1 serves the model API; Laptop 2 runs only the static UI. Full setup, Windows hotspot and
firewall steps, the `student_test` dry run, the demo-day runbook and troubleshooting are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Quick start:

```
PATHSNAP_SPLIT=student_test uvicorn api.server:app --host 0.0.0.0 --port 8000
python -m http.server 5500 --directory ui    # Laptop 2: http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
python api/smoke.py --api http://localhost:8000
```

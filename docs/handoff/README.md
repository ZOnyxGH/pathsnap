# PathSnap handoff index

PathSnap matches noisy GPS tracks to a road network: for every GPS point we generate candidate road arcs, a classifier scores each (point, arc) pair, an HMM/Viterbi decoder picks a connected sequence of arcs, and a FastAPI server + browser UI demo it on two laptops. The binding spec is `docs/PLAN.md`. If this folder and PLAN disagree, PLAN wins.

## Who owns what

| Person | Files | Handoff doc |
|---|---|---|
| Merull (lead) | `src/data.py`, `src/features.py`, `artifacts/sample/`, EDA plots, `tests/`, report integration, merges to `main` | (PLAN.md) |
| A | `src/metrics.py`, `src/train.py`, `src/hpo.py`, `artifacts/results/{folds,cv_results,...}.csv`, model plots, `artifacts/models/final.joblib` | [TEAMMATE_A_models.md](TEAMMATE_A_models.md) |
| B | `src/hmm.py`, `src/predict.py`, `src/evaluate.py`, `src/maps.py`, HMM result CSVs/plots, `artifacts/maps/*.html` | [TEAMMATE_B_hmm.md](TEAMMATE_B_hmm.md) |
| C | `api/server.py`, `ui/index.html`, two-laptop networking, README deploy section, demo + screenshots | [TEAMMATE_C_api_ui.md](TEAMMATE_C_api_ui.md) |

Interfaces between these files are fixed in PLAN section 6. To change one, agree with the owner of the other side and get Merull to commit the change to PLAN.md first.

## Checkpoints (T = start)

| Time | What must exist | Who unblocks whom |
|---|---|---|
| T+20 min | `data.py` pushed (`load_trip`, `project`). Interfaces frozen | Merull -> B (Dijkstra, maps), C (stub API) |
| T+45 min | `align_labels` + `artifacts/sample/train_sample_candidates.csv` (3 trips) | Merull -> A (metrics/train on real columns) |
| T+1 h | `build_candidates` done. B runs lam=0 HMM end to end. C wires real `predict_trip`. **First API -> UI demo works** | Merull -> B -> C |
| T+1.5 h | `cache/train_candidates.pkl` + `cache/student_test_candidates.pkl` buildable. A runs 6-model CV | Merull -> A |
| T+2.5 h | Winner chosen, HPO running. B swaps in `final.joblib` (lam=1), tunes beta | A -> B, C |
| T+3 h | Result CSVs, plots, maps committed. Two-laptop dry run with student_test trips standing in for evaluator trips | everyone |

## Setup (every laptop)

```bash
git clone <repo-url> codebase
cd codebase
pip install -r requirements.txt
```

Data is **not** in git. Expected layout (the folder that contains `codebase/`):

```
Hackathon/
  train/00000001/00000001.{track,nodes,arcs,route}
  student_test/<id>/<id>.{track,nodes,arcs,route}
  evaluator_test/...        # only on demo day, only on Laptop 1
  codebase/                 # this repo
```

If your data lives elsewhere, set `PATHSNAP_DATA` to the folder holding `train/` and `student_test/`:

```powershell
$env:PATHSNAP_DATA = "D:\pathsnap_data"      # PowerShell, current session
```

Build the candidate cache locally (takes a few minutes, written to `cache/`, never committed):

```bash
python -m src.features train
python -m src.features student_test
```

Always run code as modules from `codebase/` (`python -m src.train`, not `python src/train.py`), otherwise `from src.x import y` breaks.

Quick check: `python -m pytest tests -q`.

## Git workflow

- One branch per person: `a-models`, `b-hmm`, `c-api-ui`. Merull works on `main`.
  ```bash
  git checkout -b a-models          # once
  git add src/train.py src/metrics.py artifacts/results/cv_results.csv
  git commit -m "train: 6-model GroupKFold CV"
  git push -u origin a-models
  ```
- Open a PR into `main`; Merull merges. Pull `main` into your branch often (`git pull origin main`) so you get interface updates.
- Edit only your own files (table above). Need a change in someone else's file? Ask them.
- Never commit `cache/`, `*.pkl`, raw data (`train/`, `student_test/`, `evaluator_test/`), `__pycache__/`, or anything with credentials.
- Do commit small artifacts: result CSVs, PNG plots, folium HTML maps, `final.joblib` (must stay under 50 MB), `hpo/trials.csv`.
- `git add <specific files>`, not `git add .`, so nobody commits 300 MB of cache by accident.

## Integrity rules (non-negotiable, checked by evaluators)

- `evaluator_test` is never read by `features.build_dataset`, `train.py`, `hpo.py`, or any tuning code. Only `api/server.py` and `python -m src.evaluate --split evaluator_test` touch it, after everything is frozen.
- Model, hyperparameters, beta, sigma, lam, thresholds: chosen on train CV (and student_test for reporting/decoder tuning) only.
- No hardcoded outputs, no precomputed predictions served by the API. Every `/predict` call runs the full pipeline live, and the terminal log shows it.
- Ground truth stays server-side; the response contains only scalar metrics.
- No hardcoded trip IDs except `EXCLUDED_TRIPS = {"00000033", "00000054"}` (bad ground truth, justified in PLAN section 1).
- Every result in the report must be reproducible from a committed script + committed CSV.
- Every member must be able to explain their own part, and the overall pipeline, in the viva: why these features, why GroupKFold, why that model, how Viterbi works, why ground truth never reaches the UI.
- No sharing of code, models or results with other teams.

# Teammate C handoff: API, UI, two-laptop deployment, demo

Read `docs/PLAN.md` sections 6 (especially 6.5, 6.6), 8 and 9 once. This doc is the condensed version for your part.

## 1. Project in 5 lines

- Input per trip: a GPS track plus that trip's own road network; the pipeline outputs the road arcs the vehicle actually drove (map-matching).
- Pipeline (owned by others): candidates + features (Merull) -> classifier probabilities (A) -> HMM/Viterbi + route reconstruction (B), wrapped in one call `predict_trip(split, trip_id, bundle)`.
- **Your job:** FastAPI server on Laptop 1 that runs that pipeline live per request and scores it against server-side ground truth; a static web UI on Laptop 2 that lists the 10 evaluator trips, sends the chosen trip_id, and draws the predicted route + metrics on a Leaflet map.
- On demo day the evaluator clicks trips in the UI on Laptop 2 and then looks at the **request log in the terminal of Laptop 1**. It must work for all 10 trips.
- Hardcoded outputs, bypassing the API, or leaking ground truth to the UI are disqualifying.

## 2. Your deliverables

### `api/server.py` (contract PLAN 6.6, fixed)

| Endpoint | Response |
|---|---|
| `GET /trips` | `{"split": "evaluator_test", "trips": ["0000xxxx", ...]}` |
| `POST /predict` body `{"trip_id": "..."}` | `{"trip_id", "route_lonlat": [[lon,lat],...], "gps_lonlat": [[lon,lat],...], "metrics": {"top1", "route_f1", "route_precision", "route_recall", "continuity"}, "inference_ms"}` |
| `GET /map/{trip_id}` | folium HTML of GPS + predicted route (**no reference layer**) |
| `GET /health` (extra, harmless) | `{"ok": true, "model": bundle["name"], "split": SPLIT}` |

Skeleton (fill in, keep it this small):

```python
import os, time, datetime, joblib, numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.data import list_trips, load_trip, align_labels
from src.predict import predict_trip
from src.hmm import route_metrics
from src.metrics import point_top1_from_assignment

SPLIT = os.environ.get("PATHSNAP_SPLIT", "evaluator_test")     # "student_test" for dry runs
BUNDLE = joblib.load(os.environ.get("PATHSNAP_MODEL", "artifacts/models/final.joblib"))  # own artifact, trusted
TRIPS = list_trips(SPLIT)

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    trip = getattr(request.state, "trip_id", "-")
    print(f"[{datetime.datetime.now():%H:%M:%S}] {request.client.host}  {request.method} {request.url.path}  "
          f"{trip}  {response.status_code}  {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
    return response

class PredictIn(BaseModel):
    trip_id: str

def score(trip_id, result):
    """Ground truth is loaded ONLY here, AFTER prediction, and only scalars leave this function."""
    gt = load_trip(SPLIT, trip_id, with_route=True)
    if gt.route is None:
        return {k: None for k in ("top1", "route_f1", "route_precision", "route_recall", "continuity")}
    label_arc, _ = align_labels(gt)
    rm = route_metrics(gt, result["route"], gt.route)
    return {"top1": point_top1_from_assignment(label_arc, np.asarray(result["pred_arc"])),
            **{k: rm[k] for k in ("route_f1", "route_precision", "route_recall", "continuity")}}

@app.post("/predict")
def predict(body: PredictIn, request: Request):
    if body.trip_id not in TRIPS:           # also blocks path traversal like "../train/00000001"
        raise HTTPException(404, "unknown trip_id")
    request.state.trip_id = body.trip_id
    t0 = time.perf_counter()
    result = predict_trip(SPLIT, body.trip_id, BUNDLE)
    inference_ms = (time.perf_counter() - t0) * 1000
    gps = load_trip(SPLIT, body.trip_id, with_route=False).lonlat.tolist()
    return {"trip_id": body.trip_id, "route_lonlat": result["route_lonlat"], "gps_lonlat": gps,
            "metrics": {k: (None if v is None else float(v)) for k, v in score(body.trip_id, result).items()},
            "inference_ms": round(inference_ms, 1)}
```

Run (from `codebase/`): `uvicorn api.server:app --host 0.0.0.0 --port 8000`. Keep uvicorn's own access log on or off (`--no-access-log`); the custom line above is the one PLAN requires: `[ts] client_ip  METHOD path  trip_id  status  latency_ms`. Also print one startup line: split, number of trips, model name, model `cv_top1`.

Notes:
- `inference_ms` measures the prediction pipeline only (not GT scoring). Also forward `result["timing_ms"]` if you want a breakdown in the UI (extra key, allowed).
- Plain `def` endpoints (not `async def`) so the CPU-heavy pipeline runs in the threadpool and `/trips` stays responsive.
- `/map/{trip_id}`: validate against `TRIPS`, run `predict_trip`, return `HTMLResponse(trip_map(load_trip(SPLIT, tid, with_route=False), pred_route=result["route"]).get_root().render())` using B's `src.maps.trip_map` (agree the signature with B). Optionally cache the last result per trip_id in a dict so `/map` doesn't recompute after `/predict` (cache predictions only, never GT).
- Send `gps_lonlat` thinned if huge (up to 19k points is still fine as JSON, ~600 KB).

### `ui/index.html` (single static file, no build step)

- Leaflet from CDN: `https://unpkg.com/leaflet@1.9.4/dist/leaflet.css` and `.../leaflet.js`. **Also save both files to `ui/vendor/`** and fall back to them: the demo network may have no internet. (OSM tiles need internet too; if offline, the polylines still draw on a blank map, which is acceptable. Say so in the README.)
- API URL field at the top, default from `?api=http://192.168.x.y:8000` query param, else `localStorage`, else `http://localhost:8000`. A "Connect" button calls `GET /trips` and shows OK / error.
- Left panel: the trips from `GET /trips` as buttons (10 on demo day). Click -> `POST /predict` with `{"trip_id"}` via `fetch`, show a spinner, disable buttons while running.
- Map: GPS points as a blue polyline + small circle markers, predicted route as a red polyline (weight 4), start/end markers, `map.fitBounds(...)`, a legend.
- Metrics card: Top-1, route F1, precision, recall, continuity (as %), inference ms, trip_id, timestamp. `null` metrics shown as "n/a (no reference route)".
- Link "Open folium map" -> `<api>/map/<trip_id>` in a new tab.
- Show the HTTP error text if a request fails (don't fail silently in front of the evaluator).
- The UI never computes metrics and never fetches anything except the API.

Serve on Laptop 2 with either: double-click `ui/index.html` (CORS `*` allows the `null` origin), or `python -m http.server 5500 --directory ui` and open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`.

### README deploy section

Merull is writing `README.md` now. After it lands, add a "Deployment (two laptops)" section with: install, where to put `evaluator_test/`, the uvicorn command, firewall command, how to open the UI with `?api=`, the dry-run command with `PATHSNAP_SPLIT=student_test`, offline-tiles note, troubleshooting (below). Coordinate with Merull since README is shared.

## 3. Inputs you consume

- `predict_trip(split, trip_id, bundle, use_hmm=True) -> {"trip_id", "pred_arc": list[int], "route": list[int], "route_lonlat": [[lon,lat],...], "timing_ms": {...}}` (B, `src/predict.py`). **Before it exists** (a stub file may already be there), use your own stub in the server:
  ```python
  def predict_trip(split, trip_id, bundle, use_hmm=True):
      tr = load_trip(split, trip_id, with_route=False)
      return {"trip_id": trip_id, "pred_arc": [-1] * len(tr.t), "route": [], "route_lonlat": tr.lonlat.tolist(), "timing_ms": {}}
  ```
  and `score()` returning `None`s. Real `predict_trip` with lam=0 (`bundle=None`) arrives at T+1 h; the trained model at ~T+2.5 h.
- `load_trip`, `list_trips`, `align_labels` (Merull, `src/data.py`, available from T+20 min). `list_trips(split)` lists folders under `DATA_ROOT/split`.
- `route_metrics(trip, pred_route, ref_route)` (B, `src/hmm.py`), `point_top1_from_assignment(label_arc, pred_arc)` (A, `src/metrics.py`). Until they exist, return `None` metrics.
- `artifacts/models/final.joblib` (A). Until then start with `PATHSNAP_MODEL` unset -> wrap the load in `try/except FileNotFoundError` and pass `None` (lam=0 HMM). Never ship the demo on `None`.
- `src.maps.trip_map` (B) for `/map`.
- `DATA_ROOT` = `PATHSNAP_DATA` env var or the folder that contains `codebase/`. Splits: `train/`, `student_test/`, `evaluator_test/`, each `<id>/<id>.{track,nodes,arcs,route}`.

### evaluator_test handling

- It is handed over on demo day. Put it on **Laptop 1 only** at `DATA_ROOT/evaluator_test/<id>/<id>.track` etc. If it arrives zipped with an extra top folder, flatten it so `list_trips("evaluator_test")` returns exactly the 10 ids. Check: `python -c "from src.data import list_trips; print(list_trips('evaluator_test'))"`.
- If `.route` files are included, they are the server-side reference; the API reads them only inside `score()`. If not, metrics are `null`; the route still displays.
- Never copy it into the repo, never commit it, never run `src.features build_dataset`, training or tuning on it. No code edits after it arrives other than pointing `PATHSNAP_SPLIT` at it.
- Before demo day, dry-run with `PATHSNAP_SPLIT=student_test` (10 trips, same layout).

## 4. Step-by-step tasks

| By | Task |
|---|---|
| T+20 min | FastAPI skeleton with stub `predict_trip` (GPS returned as the route), `/trips`, `/predict`, logging middleware, CORS. `uvicorn` on 0.0.0.0 |
| T+45 min | `ui/index.html`: API URL field, trip list, POST, Leaflet drawing, metrics card, error display. Vendor Leaflet into `ui/vendor/` |
| T+45 min | Networking test with a teammate's laptop: same hotspot, `ipconfig`, firewall rule, `curl http://<ip>:8000/trips` from Laptop 2 |
| T+1 h | Wire B's real `predict_trip` (lam=0). **First end-to-end demo API -> UI on two laptops** |
| T+1.5 h | Real `score()` with `align_labels` + `route_metrics` + `point_top1_from_assignment`. `/map/{trip_id}` |
| T+2.5 h | Load `final.joblib`. Click through all 10 student_test trips; note the slowest trip's latency; report crashes to B |
| T+3 h | Two-laptop dry run with `PATHSNAP_SPLIT=student_test`. Screenshots. README deploy section |
| demo | Checklist in section 6 |

### Two-laptop networking (Windows)

1. Both laptops on the **same network**. University Wi-Fi often isolates clients, so use a phone hotspot (or Laptop 1's Mobile Hotspot) to be safe.
2. Laptop 1: `ipconfig` -> "Wireless LAN adapter Wi-Fi" -> "IPv4 Address" (e.g. `192.168.43.120`). It changes when you switch networks; re-check on demo day.
3. Laptop 1 network profile set to **Private** (Settings -> Network -> Wi-Fi -> network -> Private).
4. Firewall rule, in an **admin** PowerShell on Laptop 1:
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
   ```
   (or `netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000`). When Windows pops "allow Python to communicate", tick Private and allow.
5. Start the API on Laptop 1 bound to all interfaces: `uvicorn api.server:app --host 0.0.0.0 --port 8000`.
6. Laptop 1 self-test: `curl http://localhost:8000/trips`, then `curl http://<LAPTOP1_IP>:8000/trips`.
7. Laptop 2: open `http://<LAPTOP1_IP>:8000/trips` in a browser, then the UI with `?api=http://<LAPTOP1_IP>:8000`.
8. Troubleshooting: timeout -> firewall/profile or client isolation (switch to hotspot); connection refused -> server bound to 127.0.0.1 or not running; CORS error in browser console -> middleware missing; 404 on every trip -> wrong `PATHSNAP_SPLIT`/data path.

## 5. Report sections you own

### Section 7: API, UI and deployment

Required by the brief:
- Screenshots of the working UI.
- Screenshots of successful API responses **on the CMD of the receiving laptop** (Laptop 1 terminal showing the request log lines with client IP of Laptop 2, trip_id, status 200, latency).
- The submitted source code must match what is demonstrated.

Screenshots to capture during the T+3 h dry run (save to `artifacts/plots/screens/` or hand to Merull):
1. UI with the trip list loaded (API URL showing Laptop 1's IP).
2. UI after predicting 2-3 different trips: map with GPS + predicted route, metrics card visible.
3. Laptop 1 terminal with several request log lines from Laptop 2's IP (`GET /trips`, `POST /predict` x several).
4. One raw JSON response (`/docs` Swagger "Try it out", or `curl -X POST http://<ip>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\":\"00000007\"}"`), with the long arrays collapsed.
5. The `/map/{trip_id}` folium page.
6. Optional: `ipconfig` + firewall rule output, to document the setup.

Write a short paragraph on: architecture (Laptop 1 = model + HMM + network + GT scoring; Laptop 2 = static UI), how ground truth stays server-side (loaded only in `score()` after prediction, only scalars returned), how CORS/firewall/binding were configured, measured latency per trip.

## 6. Definition of done + verification

- [ ] `uvicorn api.server:app --host 0.0.0.0 --port 8000` starts with `PATHSNAP_SPLIT=student_test` and prints split/model info.
- [ ] `curl http://localhost:8000/trips` lists 10 ids; `POST /predict` returns 200 with all keys for **every** one of the 10 trips (loop over them once with a small script or the UI).
- [ ] `POST /predict` with an unknown id or `"../train/00000001"` returns 404.
- [ ] Response contains no arc ids of the reference route, no `label_arc`, no `.route` content: only the four scalar route metrics + top1.
- [ ] Terminal shows exactly one log line per request in the PLAN format, including Laptop 2's IP.
- [ ] UI works from Laptop 2 via IP; API URL editable without editing code; errors visible.
- [ ] UI works with internet off (vendored Leaflet; tiles may be blank).
- [ ] `/map/{trip_id}` renders GPS + predicted route, no reference layer.
- [ ] README deploy section committed.

### Demo-day checklist

1. Laptop 1: `git pull`, `pip install -r requirements.txt`, `final.joblib` present (`python -c "import joblib; print(joblib.load('artifacts/models/final.joblib')['name'])"`).
2. Copy `evaluator_test/` next to `train/` (or set `PATHSNAP_DATA`), verify `list_trips('evaluator_test')` = 10 ids.
3. Connect both laptops to the hotspot; `ipconfig`; confirm Private profile and firewall rule.
4. `$env:PATHSNAP_SPLIT = "evaluator_test"`; start uvicorn; keep the terminal visible and large font.
5. Warm-up: predict one trip yourself (first call loads things; the evaluator should not wait on it).
6. Laptop 2: open the UI with `?api=http://<ip>:8000`, connect, show 10 trips.
7. Let the evaluator pick; point at the Laptop 1 log line each time.
8. Fallback: if Wi-Fi fails, both on the phone hotspot; last resort run the UI on Laptop 1 against `localhost` and explain (but two laptops is the requirement).
9. Be ready to answer: "where is the ground truth loaded?" (show `score()`), "is anything precomputed?" (no; show the log latency and `predict_trip`).

## 7. Gotchas

- **Trip ids are zero-padded strings** ("00000007"). Never cast to int in JS or Python.
- **Worldwide coordinates**: trips are in Germany, Moscow, USA, Iraq, Spain... Never hardcode a map centre; always `fitBounds` on the returned GPS. Arrays are `[lon, lat]`; Leaflet wants `[lat, lon]`. Swap them, or every route lands in the ocean.
- **Latency**: B targets < 5 s per typical trip, but the longest tracks (19k points, networks of 155k nodes) can take longer. Show a spinner, set no short `fetch` timeout, and do the warm-up call.
- **Huge GPS gaps** (trips with 52,810 s / 153,064 s gaps exist in train): the predicted route may have breaks; the polyline will draw a straight jump. That is expected; continuity < 1 shows it.
- **Bad-GT trips 33 and 54** are train trips; they won't appear in the demo, but if you test on train, low metrics there are a ground-truth problem, not a bug.
- **Reverse twins**: a road's two directions are separate arcs with identical geometry, so a wrong-direction prediction looks perfect on the map but scores low on route F1. Expect that question.
- **Leakage**: the server must not use the GT for prediction. `predict_trip` loads `with_route=False`; GT is read only in `score()`, after prediction.
- **`print` buffering**: use `flush=True`, or the log lines appear late in the terminal while the evaluator watches.
- **Run from `codebase/`** so `src` and `api` import and relative `artifacts/...` paths resolve. Add an empty `api/__init__.py` only if uvicorn can't import `api.server` (Python 3 namespace packages usually work without it).
- **Firewall + hotspot names change** between rehearsal and demo; redo steps 2-4 of the networking list on the day.

## 8. Who to talk to

- **B**: `predict_trip` return dict, speed per trip, `trip_map` signature for `/map`, `route_metrics`. Report any trip that raises an exception immediately.
- **A**: `final.joblib` location/size, `point_top1_from_assignment`.
- **Merull**: `load_trip`/`list_trips`/`DATA_ROOT`, `align_labels`, README (shared file), evaluator_test placement, report section 7 integration.

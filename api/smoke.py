"""End-to-end smoke test for the PathSnap API. Stdlib only.

Usage: python api/smoke.py [--api http://localhost:8000] [--trip ID] [--timeout 120]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

REQUIRED = ("trip_id", "route_lonlat", "gps_lonlat", "metrics", "inference_ms")
METRIC_KEYS = ("top1", "route_f1", "route_precision", "route_recall", "continuity")
FORBIDDEN = ("route", "pred_arc", "label_arc", "ref_route", "reference_route", "gt_route")
BAD_IDS = ("does-not-exist", "../train/00000001")


def call(method, url, payload=None, timeout=120.0):
    """Return (status, body_text); status 0 means the connection failed."""
    data, headers = None, {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        return 0, f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_lonlat(value, name, errors):
    """Validate a list of [lon, lat] pairs; return the pair count."""
    if not isinstance(value, list):
        errors.append(f"{name} is not a list")
        return 0
    bad = sum(1 for pair in value
              if not (isinstance(pair, list) and len(pair) == 2
                      and all(is_number(v) for v in pair)))
    if bad:
        errors.append(f"{name} has {bad} malformed [lon,lat] entries")
    return len(value)


def check_metrics(value, errors):
    """Validate the metrics object: exactly five keys, numbers or None."""
    if not isinstance(value, dict):
        errors.append("metrics is not an object")
        return
    missing = [k for k in METRIC_KEYS if k not in value]
    extra = [k for k in value if k not in METRIC_KEYS]
    if missing:
        errors.append(f"metrics missing {missing}")
    if extra:
        errors.append(f"metrics has unexpected keys {extra}")
    bad = [k for k, v in value.items() if v is not None and not is_number(v)]
    if bad:
        errors.append(f"metrics values not number/None: {bad}")


def check_predict(body, trip_id):
    """Validate one /predict response. Returns (errors, route_pts, gps_pts, metrics, stub)."""
    errors = []
    if not isinstance(body, dict):
        return ["response is not a JSON object"], 0, 0, {}, None
    for key in REQUIRED:
        if key not in body:
            errors.append(f"missing key {key!r}")
    leaked = [k for k in FORBIDDEN if k in body]
    if leaked:
        errors.append(f"leaked ground-truth/prediction keys: {leaked}")
    if body.get("trip_id") != trip_id:
        errors.append(f"trip_id is {body.get('trip_id')!r}, expected {trip_id!r}")
    route_pts = check_lonlat(body.get("route_lonlat"), "route_lonlat", errors)
    gps_pts = check_lonlat(body.get("gps_lonlat"), "gps_lonlat", errors)
    check_metrics(body.get("metrics"), errors)
    if not is_number(body.get("inference_ms")):
        errors.append("inference_ms is not a number")
    stub = body.get("stub")
    if stub is not None and not isinstance(stub, bool):
        errors.append("stub is not a boolean")
    metrics = body.get("metrics") if isinstance(body.get("metrics"), dict) else {}
    return errors, route_pts, gps_pts, metrics, stub


def fmt(value, digits=4):
    """Metric for the table: short float or 'n/a'."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a running PathSnap API.")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--trip", default=None, help="test only this trip id")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout (s)")
    args = parser.parse_args()
    api = args.api.rstrip("/")

    failures = []

    status, text = call("GET", f"{api}/health", timeout=args.timeout)
    if status != 200:
        print(f"FAIL /health -> HTTP {status}: {text[:300]}")
        print("\nFAIL: API is not reachable; is uvicorn running?")
        return 1
    try:
        health = json.loads(text)
    except json.JSONDecodeError:
        health = {}
    print(f"health: ok={health.get('ok')}  model={health.get('model')}  "
          f"split={health.get('split')}  trips={health.get('trips')}  "
          f"stub={health.get('stub')}")

    status, text = call("GET", f"{api}/trips", timeout=args.timeout)
    split = "?"
    try:
        listing = json.loads(text)
        trips = listing.get("trips", [])
        split = listing.get("split", "?")
    except (json.JSONDecodeError, AttributeError):
        trips = []
    if status != 200:
        failures.append(f"GET /trips -> HTTP {status}")
    if not isinstance(trips, list) or not all(isinstance(t, str) for t in trips):
        failures.append("GET /trips did not return a list of strings")
        trips = []
    elif not trips:
        failures.append("GET /trips returned no trips")
    print(f"trips: split={split}  count={len(trips)}")

    targets = [args.trip] if args.trip else trips
    if args.trip and args.trip not in trips:
        failures.append(f"--trip {args.trip!r} is not in GET /trips")

    passed = 0
    for trip_id in targets:
        status, text = call("POST", f"{api}/predict", {"trip_id": trip_id}, args.timeout)
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = None
        errors, route_pts, gps_pts, metrics, stub = check_predict(body, trip_id)
        if status != 200:
            errors.insert(0, f"HTTP {status}: {text[:200]}")
        ok = not errors
        passed += 1 if ok else 0
        ms = body.get("inference_ms") if isinstance(body, dict) else None
        print(f"{trip_id}  status={status}  n_route_pts={route_pts}  n_gps_pts={gps_pts}  "
              f"top1={fmt(metrics.get('top1'))}  route_f1={fmt(metrics.get('route_f1'))}  "
              f"inference_ms={fmt(ms, 1)}  "
              f"stub={'yes' if stub else 'no'}  {'PASS' if ok else 'FAIL'}")
        for error in errors:
            print(f"    ! {error}")
            failures.append(f"{trip_id}: {error}")

    for bad_id in BAD_IDS:
        status, text = call("POST", f"{api}/predict", {"trip_id": bad_id}, args.timeout)
        ok = status == 404
        print(f"404 check: trip_id={bad_id!r}  status={status}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"trip_id {bad_id!r} returned HTTP {status}, expected 404")

    print(f"\n{'PASS' if not failures else 'FAIL'}: {passed}/{len(targets)} trips passed, "
          f"{len(BAD_IDS)} 404 checks, {len(failures)} failure(s)")
    for failure in failures:
        print(f"  - {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

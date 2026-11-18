"""Runnable with `python -m pytest` or `python tests/test_data.py` from codebase/."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DATA_ROOT, EXCLUDED_TRIPS, Trip, align_labels, list_trips, load_trip, seg_dist  # noqa: E402
from src.features import FEATURES, build_candidates  # noqa: E402


def test_seg_dist():
    d, t = seg_dist(np.array([[5.0, 3.0], [-4.0, 3.0], [12.0, 0.0]]), np.array([0.0, 0.0]), np.array([10.0, 0.0]))
    assert np.allclose(d, [3.0, 5.0, 2.0]) and np.allclose(t, [0.5, 0.0, 1.0])


def test_align_labels_monotonic():
    # out along y=0, back along y=20; the 4th point sits closer to the return road but travel order says outbound
    nodes = np.array([[0, 0], [100, 0], [200, 0], [300, 0], [300, 20], [200, 20], [100, 20], [0, 20]], float)
    arcs = np.c_[np.arange(7), np.arange(1, 8)]
    xy = np.array([[10, 0], [50, 0], [110, 0], [150, 12], [250, 0], [250, 20], [210, 20], [150, 20],
                   [90, 20], [50, 20]], float)
    trip = Trip("syn", np.arange(len(xy), dtype=float), xy, xy, (0, 0), nodes, nodes, arcs,
                np.full(7, 100.0), np.arange(7))
    lab, dist = align_labels(trip)
    assert lab.tolist() == [0, 0, 1, 1, 2, 4, 4, 5, 6, 6]
    assert dist[3] == 12 and (np.diff(lab) >= 0).all()


def test_align_labels_uturn_twin():
    # out-and-back on one road with reverse twin arcs: equal distances, only heading tells the legs apart
    nodes = np.array([[0, 0], [100, 0], [200, 0]], float)
    arcs = np.array([[0, 1], [1, 2], [2, 1], [1, 0]])
    xy = np.array([[20, 0], [70, 0], [130, 0], [180, 0], [170, 0], [130, 0], [70, 0], [20, 0]], float)
    trip = Trip("uturn", np.arange(len(xy)) * 10.0, xy, xy, (0, 0), nodes, nodes, arcs,
                np.full(4, 100.0), np.arange(4))
    assert align_labels(trip)[0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_build_candidates_real_trip():
    root = DATA_ROOT / "train"
    tid = min((t for t in list_trips("train") if t not in EXCLUDED_TRIPS),
              key=lambda t: (root / t / f"{t}.track").stat().st_size)
    trip = load_trip("train", tid, with_route=False)
    assert trip.route is None
    c = build_candidates(trip)
    assert list(c.columns) == ["trip_id", "pt_idx", "arc_id"] + FEATURES
    assert c["pt_idx"].dtype == np.int32 and c["arc_id"].dtype == np.int64
    assert all(c[f].dtype == np.float32 for f in FEATURES)
    assert (c["dist"] <= 200).all()
    key = c["pt_idx"].to_numpy().astype(np.float64) * 1e4 + c["dist"].to_numpy() / 1e3
    assert (np.diff(key) >= 0).all()  # sorted by (pt_idx, dist)
    assert c.groupby("pt_idx").size().max() <= 16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

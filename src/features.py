"""Candidate generation, features and dataset cache (PLAN sections 4 and 6.2). Owner: Merull (lead)."""
import sys
import time
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import EXCLUDED_TRIPS, Trip, align_labels, bearing, kinematics, list_trips, load_trip, seg_dist

RADIUS, K, FALLBACK_RADIUS, FALLBACK_K = 50.0, 16, 200.0, 4
STEP = 25.0  # arc densification step for the KD-tree
FEATURES = ["dist", "dist_rank", "dist_gap", "dist_ratio", "proj_frac", "edge_proj", "arc_len",
            "bearing_diff", "cos_bearing", "heading_ok", "speed", "dt_prev", "dt_next", "n_cand",
            "has_twin", "out_deg", "in_deg", "prev_link", "next_link", "fallback"]
ROOT = Path(__file__).resolve().parents[1]


def _pairs(xy, n_arcs, A, B, tree, aid, radius, k):
    """(pt, arc, dist, t) of the k nearest arcs within radius for each point, sorted by (pt, dist, arc)."""
    # a sample is at most STEP/2 from any part of its sub-segment, hence the padding
    hits = tree.query_ball_point(xy, radius + STEP / 2 + 1, return_sorted=False)
    cnt = np.fromiter(map(len, hits), np.int64, len(xy))
    pt = np.repeat(np.arange(len(xy)), cnt)
    arc = aid[np.fromiter(chain.from_iterable(hits), np.int64, cnt.sum())]
    u = np.unique(pt * n_arcs + arc)  # one row per (pt, arc)
    pt, arc = u // n_arcs, u % n_arcs
    d, t = seg_dist(xy[pt], A[arc], B[arc])
    keep = d <= radius
    pt, arc, d, t = pt[keep], arc[keep], d[keep], t[keep]
    o = np.lexsort((arc, d, pt))
    pt, arc, d, t = pt[o], arc[o], d[o], t[o]
    first = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
    rank = np.arange(len(pt)) - np.repeat(first, np.diff(np.r_[first, len(pt)]))
    keep = rank < k
    return pt[keep], arc[keep], d[keep], t[keep]


def build_candidates(trip: Trip) -> pd.DataFrame:
    n, M, N = len(trip.xy), len(trip.arcs), len(trip.nodes_xy)
    frm, to = trip.arcs[:, 0], trip.arcs[:, 1]
    A, B = trip.nodes_xy[frm], trip.nodes_xy[to]
    # densify so that long arcs (up to km) are found by the KD-tree
    nseg = np.maximum(1, np.ceil(trip.arc_len / STEP)).astype(np.int64)
    aid = np.repeat(np.arange(M), nseg)
    f = (np.arange(len(aid)) - np.repeat(np.cumsum(nseg) - nseg, nseg) + 0.5) / np.repeat(nseg, nseg)
    tree = cKDTree(A[aid] + (B - A)[aid] * f[:, None])

    pt, arc, d, t = _pairs(trip.xy, M, A, B, tree, aid, RADIUS, K)
    fb = np.zeros(len(pt), bool)
    miss = np.setdiff1d(np.arange(n), pt)
    if len(miss):
        p2, a2, d2, t2 = _pairs(trip.xy[miss], M, A, B, tree, aid, FALLBACK_RADIUS, FALLBACK_K)
        pt, arc = np.r_[pt, miss[p2]], np.r_[arc, a2]
        d, t, fb = np.r_[d, d2], np.r_[t, t2], np.r_[fb, np.ones(len(p2), bool)]
        o = np.lexsort((arc, d, pt))
        pt, arc, d, t, fb = pt[o], arc[o], d[o], t[o], fb[o]

    R = len(pt)
    first = np.r_[0, np.flatnonzero(np.diff(pt)) + 1]
    counts = np.diff(np.r_[first, R])
    n_cand = np.repeat(counts, counts)
    rank = np.arange(R) - np.repeat(first, counts)
    dmin = np.repeat(d[first], counts)

    speed, heading = kinematics(trip.xy, trip.t)
    ok = speed >= 1.0
    dt = np.diff(trip.t) if n > 1 else np.zeros(1)
    dt_prev, dt_next = np.r_[dt[:1], dt], np.r_[dt, dt[-1:]]  # ends reuse the only neighbour

    bd = np.abs((heading[pt] - bearing(B - A)[arc] + 180) % 360 - 180)
    bd = np.where(ok[pt], bd, 90.0)

    # reverse twin among the point's candidates
    key = pt * N * N + frm[arc] * N + to[arc]
    has_twin = np.isin(pt * N * N + to[arc] * N + frm[arc], key)

    # 1-hop connectivity to the 3 nearest candidates of the neighbouring points
    top3 = np.full((n + 2, 3), -1, np.int64)  # rows 0 and n+1 are empty padding for pt-1 / pt+1
    sel = rank < 3
    top3[pt[sel] + 1, rank[sel]] = arc[sel]

    def link(shift, prev):
        c = top3[pt + 1 + shift]
        hop = (to[np.maximum(c, 0)] == frm[arc][:, None]) if prev else (frm[np.maximum(c, 0)] == to[arc][:, None])
        return (((c == arc[:, None]) | hop) & (c >= 0)).any(1)

    arc_len = trip.arc_len[arc]
    out = pd.DataFrame({
        "trip_id": trip.trip_id, "pt_idx": pt.astype(np.int32), "arc_id": arc.astype(np.int64),
        "dist": d, "dist_rank": rank, "dist_gap": d - dmin, "dist_ratio": d / (dmin + 1),
        "proj_frac": t, "edge_proj": np.minimum(t, 1 - t) * arc_len, "arc_len": arc_len,
        "bearing_diff": bd, "cos_bearing": np.cos(np.radians(bd)), "heading_ok": ok[pt],
        "speed": speed[pt], "dt_prev": dt_prev[pt], "dt_next": dt_next[pt], "n_cand": n_cand,
        "has_twin": has_twin, "out_deg": np.bincount(frm, minlength=N)[to[arc]],
        "in_deg": np.bincount(to, minlength=N)[frm[arc]],
        "prev_link": link(-1, True), "next_link": link(1, False), "fallback": fb,
    })
    out[FEATURES] = out[FEATURES].astype(np.float32)
    return out


def add_labels(cands: pd.DataFrame, trip: Trip) -> pd.DataFrame:
    if trip.route is None:
        raise ValueError(f"trip {trip.trip_id} has no route; cannot add labels")
    label_arc, _ = align_labels(trip)
    cands = cands.copy()
    la = label_arc[cands["pt_idx"].to_numpy()]
    cands["label"] = (cands["arc_id"].to_numpy() == la).astype(np.int8)
    cands["label_arc"] = la
    return cands


def _cache(split: str) -> Path:
    return ROOT / "cache" / f"{split}_candidates.pkl"


def build_dataset(split: str) -> pd.DataFrame:
    if split == "evaluator_test":
        raise ValueError("evaluator_test must never be built into a cache (PLAN section 8)")
    parts, n_pts = [], 0
    for tid in list_trips(split):
        if split == "train" and tid in EXCLUDED_TRIPS:
            continue
        t0 = time.time()
        trip = load_trip(split, tid)
        c = build_candidates(trip)
        if trip.route is not None:
            c = add_labels(c, trip)
        parts.append(c)
        n_pts += len(trip.t)
        print(f"{split} {tid} pts={len(trip.t)} rows={len(c)} {time.time() - t0:.1f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.attrs["n_pts"] = n_pts
    _cache(split).parent.mkdir(exist_ok=True)
    df.to_pickle(_cache(split))
    return df


def load_dataset(split: str) -> pd.DataFrame:
    return pd.read_pickle(_cache(split)) if _cache(split).exists() else build_dataset(split)


def _write_sample(df: pd.DataFrame, split: str, max_rows=40000, n_trips=3):
    chosen, total = [], 0
    for tid, r in df.groupby("trip_id").size().items():
        if total + r <= max_rows:
            chosen.append(tid)
            total += r
        if len(chosen) == n_trips:
            break
    out = ROOT / "artifacts" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    df[df["trip_id"].isin(chosen)].to_csv(out / f"{split}_sample_candidates.csv", index=False,
                                          float_format="%.5g")
    print("sample trips:", chosen)


def _summary(df: pd.DataFrame, split: str):
    n_pts = df.attrs["n_pts"]
    pts = df.drop_duplicates(["trip_id", "pt_idx"])
    n_lab = (pts["label_arc"] >= 0).sum()
    hit = (df["label"] == 1).sum()  # at most one true row per point
    lab = df[df["label_arc"] >= 0]
    near = lab[lab["dist_rank"] == 0]
    score = lab["dist"] + 30 * (lab["bearing_diff"] > 90) + 10 * lab["bearing_diff"] / 180
    head = lab.assign(s=score).sort_values(["trip_id", "pt_idx", "s"]).drop_duplicates(["trip_id", "pt_idx"])
    print(f"== {split}: trips={df['trip_id'].nunique()} points={n_pts} rows={len(df)} "
          f"labelled_points={n_lab} labelled_frac={n_lab / n_pts:.4f}")
    print(f"candidate miss rate={1 - hit / n_lab:.4%}  "
          f"nearest top1={(near['arc_id'] == near['label_arc']).mean():.4f}  "
          f"nearest+heading top1={(head['arc_id'] == head['label_arc']).mean():.4f}")


if __name__ == "__main__":
    split = sys.argv[1]
    t0 = time.time()
    df = build_dataset(split)
    print(f"built in {time.time() - t0:.0f}s")
    _write_sample(df, split)
    if "label" in df:
        _summary(df, split)

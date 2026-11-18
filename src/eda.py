"""EDA for the report: inventory, summary, plots, Folium maps. Run: python -m src.eda
Partitions used: train (80 trips) and student_test (10 trips) only."""
import time

import branca.element as be
import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

from .data import DATA_ROOT, EXCLUDED_TRIPS, align_labels, list_trips, load_trip
from .features import ROOT, load_dataset

SPLITS = ["train", "student_test"]
RES, PLOTS, MAPS = ROOT / "artifacts" / "results", ROOT / "artifacts" / "plots", ROOT / "artifacts" / "maps"
PAL = {"train": "#1f77b4", "student_test": "#e8710a"}
BAD = "00000033"


def trip_row(split, tid):
    """One inventory row + per-point arrays for a trip."""
    base = DATA_ROOT / split / tid / tid
    sizes = {f"size_{e}": (base.with_suffix("." + e).stat().st_size if base.with_suffix("." + e).exists() else 0)
             for e in ("track", "nodes", "arcs", "route")}
    tr = load_trip(split, tid)
    n, N, M = len(tr.t), len(tr.nodes_xy), len(tr.arcs)
    lon, lat = tr.lonlat[:, 0], tr.lonlat[:, 1]
    dt = np.diff(tr.t)
    ok = (dt > 0) & (dt <= 10)
    step = np.linalg.norm(np.diff(tr.xy, axis=0), axis=1)
    speed = step[ok] / dt[ok]
    frm, to = tr.arcs[:, 0], tr.arcs[:, 1]
    comp = connected_components(coo_matrix((np.ones(M), (frm, to)), shape=(N, N)), directed=True,
                                connection="weak")[1]
    deg = np.bincount(np.r_[frm, to], minlength=N)
    r = tr.route
    lab, ld = align_labels(tr)
    nb = tr.nodes_lonlat
    row = dict(
        split=split, trip_id=tid, n_points=n, duration_s=tr.t[-1] - tr.t[0], n_nodes=N, n_arcs=M,
        route_len=len(r), **sizes,
        n_nan_coords=int(np.isnan(tr.lonlat).any(1).sum()) + int(np.isnan(nb).any(1).sum()),
        n_invalid_coords=int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum()),
        n_bad_ids=int((tr.arcs < 0).sum() + (tr.arcs >= N).sum() + (r < 0).sum() + (r >= M).sum()),
        n_nonmono_t=int((dt <= 0).sum()), n_gaps_10s=int((dt > 10).sum()), max_gap_s=float(dt.max()),
        median_speed_mps=float(np.median(speed)) if len(speed) else np.nan,
        extent_km=float(np.hypot(*np.ptp(tr.xy, axis=0)) / 1000),
        net_extent_km=float(np.hypot(*np.ptp(tr.nodes_xy, axis=0)) / 1000),
        n_components=int(comp.max() + 1), largest_comp_frac=float(np.bincount(comp).max() / N),
        n_isolated_nodes=int((deg == 0).sum()),
        route_contig_frac=float((to[r[:-1]] == frm[r[1:]]).mean()) if len(r) > 1 else np.nan,
        label_dist_p50=float(np.median(ld)), label_dist_p90=float(np.percentile(ld, 90)),
        labelled_frac=float((lab >= 0).mean()), off50_frac=float((ld > 50).mean()),
        lon0=tr.origin[1], lat0=tr.origin[0])
    return row, dict(dt=dt, speed=speed, ld=ld, lab=lab)


def summarise(inv, cand):
    rows = []
    for name, d in [("train", inv[inv.split == "train"]),
                    ("train_clean", inv[(inv.split == "train") & ~inv.trip_id.isin(EXCLUDED_TRIPS)]),
                    ("student_test", inv[inv.split == "student_test"])]:
        sp = "train" if name == "train_clean" else name
        c = cand[sp]
        if name == "train_clean":
            c = c[~c.trip_id.isin(EXCLUDED_TRIPS)]  # cache is already clean; kept for safety
        pts = c.drop_duplicates(["trip_id", "pt_idx"])
        lab = pts[pts.label_arc >= 0]
        hit = c[c.label == 1]
        rows.append(dict(
            partition=name, n_trips=len(d), total_points=d.n_points.sum(),
            points_median=d.n_points.median(), points_min=d.n_points.min(), points_max=d.n_points.max(),
            duration_s_median=d.duration_s.median(), nodes_median=d.n_nodes.median(), arcs_median=d.n_arcs.median(),
            route_len_median=d.route_len.median(), extent_km_median=d.extent_km.median(),
            median_speed_mps_median=d.median_speed_mps.median(), speed_min=d.median_speed_mps.min(),
            speed_max=d.median_speed_mps.max(),
            nan_or_invalid_coords=int(d.n_nan_coords.sum() + d.n_invalid_coords.sum()), bad_ids=int(d.n_bad_ids.sum()),
            nonmono_timestamps=int(d.n_nonmono_t.sum()), gaps_10s_total=int(d.n_gaps_10s.sum()),
            gaps_10s_per_trip=d.n_gaps_10s.mean(), max_gap_s=d.max_gap_s.max(),
            trips_multi_component=int((d.n_components > 1).sum()), largest_comp_frac_median=d.largest_comp_frac.median(),
            isolated_nodes_total=int(d.n_isolated_nodes.sum()),
            trips_route_noncontiguous=int((d.route_contig_frac < 1).sum()),
            route_contig_frac_min=d.route_contig_frac.min(), route_contig_frac_mean=d.route_contig_frac.mean(),
            label_dist_p50_median=d.label_dist_p50.median(), label_dist_p90_median=d.label_dist_p90.median(),
            labelled_frac_pooled=float((d.labelled_frac * d.n_points).sum() / d.n_points.sum()),
            off50_frac_pooled=float((d.off50_frac * d.n_points).sum() / d.n_points.sum()),
            cand_points=len(pts), cand_n_median=pts.n_cand.median(), cand_n_p95=pts.n_cand.quantile(.95),
            cand_fallback_frac=float(pts.fallback.mean()),
            cand_miss_rate=1 - len(hit) / max(len(lab), 1),
            cand_recall_25m=float((hit.dist <= 25).sum() / max(len(lab), 1)),
            cand_recall_50m=float((hit.dist <= 50).sum() / max(len(lab), 1))))
    return pd.DataFrame(rows)


def plot_all(inv, arr, cand):
    sns.set_theme(style="whitegrid")
    PLOTS.mkdir(parents=True, exist_ok=True)
    part = "partitions: train (80 trips) + student_test (10 trips)"

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(PLOTS / f"eda_{name}.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(inv, x="n_points", hue="split", log_scale=True, bins=30, palette=PAL, ax=ax)
    ax.set(title=f"GPS points per trip ({part})", xlabel="points per trip (log)", ylabel="trips")
    save(fig, "points_per_trip")

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        dt = np.clip(np.concatenate([arr[s, t]["dt"] for t in list_trips(s)]), 0, 61)
        v, c = np.unique(dt.round().astype(int), return_counts=True)
        ax.plot(v, c / c.sum(), "o-", ms=3, color=PAL[s], label=s)
    ax.set(yscale="log", xlabel="sampling interval dt (s); 61 = all steps > 60 s", ylabel="fraction of steps (log)",
           title=f"Sampling interval ({part})")
    ax.legend()
    save(fig, "sampling_interval")

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        v = np.concatenate([arr[s, t]["speed"] for t in list_trips(s)])
        sns.histplot(np.clip(v, 0, 45), bins=60, stat="density", element="step", fill=False, color=PAL[s], label=s, ax=ax)
    ax.set(xlabel="speed between consecutive fixes, dt<=10 s (m/s, clipped at 45)", title=f"Speed distribution ({part})")
    ax.legend()
    save(fig, "speed")

    fig, ax = plt.subplots(figsize=(7, 4))
    for s in SPLITS:
        v = np.clip(np.concatenate([arr[s, t]["ld"] for t in list_trips(s)]), 0, 200)
        sns.histplot(v, bins=100, stat="density", element="step", fill=False, color=PAL[s], label=s, ax=ax)
    ax.axvline(50, color="red", ls="--", label="50 m cap (label rule)")
    ax.set(yscale="log", xlabel="distance to DP-aligned route arc (m, clipped at 200)",
           title=f"GPS-to-route distance, all points ({part})")
    ax.legend()
    save(fig, "gps_to_route_dist")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for s in SPLITS:
        pts = cand[s].drop_duplicates(["trip_id", "pt_idx"])
        sns.histplot(pts.n_cand, bins=np.arange(0.5, 17.5), stat="density", element="step", fill=False,
                     color=PAL[s], label=s, ax=a1)
        nl = (pts.label_arc >= 0).sum()
        d = cand[s].loc[cand[s].label == 1, "dist"].to_numpy()
        r = np.arange(0, 51)
        a2.plot(r, [(d <= x).sum() / nl for x in r], color=PAL[s], label=s)
    a1.set(xlabel="candidates per point (cap K=16)", title="Candidates per point")
    a1.legend()
    a2.set(xlabel="radius (m)", ylabel="recall of labelled arc", title="Candidate recall vs radius")
    a2.legend()
    fig.suptitle(f"Candidate table, cached ({part}; train excludes bad-GT trips 33, 54)")
    save(fig, "candidates")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(inv, x="lon0", y="lat0", hue="split", palette=PAL, s=40, ax=ax)
    ax.set(xlabel="lon (mean of track)", ylabel="lat (mean of track)", title=f"Trip origins ({part})")
    save(fig, "origins")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(inv, x="n_points", y="n_arcs", hue="split", palette=PAL, size="extent_km", ax=ax)
    ax.set(xscale="log", yscale="log", xlabel="track length (points)", ylabel="network size (arcs)",
           title=f"Network size vs track length ({part})")
    save(fig, "network_vs_track")

    tot = inv.groupby("split")[["n_points"]].sum()["n_points"]
    pt_cols = {"NaN/invalid coords": inv.n_nan_coords + inv.n_invalid_coords, "non-increasing timestamp": inv.n_nonmono_t,
               "step gap > 10 s": inv.n_gaps_10s, "point > 50 m from labelled arc": inv.off50_frac * inv.n_points}
    pt = pd.DataFrame({k: v.groupby(inv.split).sum() / tot * 100 for k, v in pt_cols.items()}).T
    tc = {"no .route file": inv.size_route == 0, "network > 1 component": inv.n_components > 1,
          "isolated nodes": inv.n_isolated_nodes > 0, "route not contiguous": inv.route_contig_frac < 1,
          "has gap > 10 s": inv.n_gaps_10s > 0, "> 5% points unlabelled": inv.off50_frac > .05}
    tcd = pd.DataFrame({k: v.groupby(inv.split).mean() * 100 for k, v in tc.items()}).T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    pt[SPLITS].plot.barh(ax=a1, color=[PAL[s] for s in SPLITS])
    a1.set(xlabel="% of points", title="Point-level issues")
    tcd[SPLITS].plot.barh(ax=a2, color=[PAL[s] for s in SPLITS])
    a2.set(xlabel="% of trips affected", title="Trip-level issues")
    fig.suptitle(f"Missing / invalid values ({part})")
    save(fig, "missing_invalid")


def traj_plot(split, tid, name, title_extra=""):
    tr = load_trip(split, tid)
    lab, ld = align_labels(tr)
    A, B = tr.nodes_xy[tr.arcs[:, 0]], tr.nodes_xy[tr.arcs[:, 1]]
    rs = np.r_[A[tr.route], B[tr.route]]
    pts = np.r_[tr.xy, rs]
    lo, hi = pts.min(0) - 200, pts.max(0) + 200
    inb = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1)
    seg = np.stack([A[inb], B[inb]], 1)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(LineCollection(seg, colors="#bbbbbb", linewidths=0.5, label=f"network ({inb.sum()} arcs)"))
    ax.add_collection(LineCollection(np.stack([A[tr.route], B[tr.route]], 1), colors="#2ca02c", linewidths=3,
                                     label="reference route", zorder=4))
    far = ld > 50
    ax.scatter(*tr.xy[~far].T, s=3, c="#1f77b4", label=f"GPS, <= 50 m from route ({(~far).sum()})", zorder=3)
    ax.scatter(*tr.xy[far].T, s=8, c="#d62728", label=f"GPS, > 50 m from route ({far.sum()})", zorder=5)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="x (m)", ylabel="y (m)",
           title=f"{split} trip {tid}{title_extra}: median GPS-to-route {np.median(ld):.0f} m, "
                 f"{(ld > 50).mean():.0%} points > 50 m")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOTS / f"eda_{name}.png", dpi=120)
    plt.close(fig)


def folium_map(split, tid, max_pts=1200, max_arcs=15000):
    tr = load_trip(split, tid)
    lab, ld = align_labels(tr)
    A, B = tr.nodes_xy[tr.arcs[:, 0]], tr.nodes_xy[tr.arcs[:, 1]]
    lo, hi = tr.xy.min(0) - 200, tr.xy.max(0) + 200
    inb = ((np.minimum(A, B) <= hi) & (np.maximum(A, B) >= lo)).all(1) & (tr.arcs[:, 0] < tr.arcs[:, 1])  # twins once
    ids = np.flatnonzero(inb)
    if len(ids) > max_arcs:
        ids = np.sort(np.random.default_rng(0).choice(ids, max_arcs, replace=False))
    ll = lambda a: [[round(float(y), 5), round(float(x), 5)] for x, y in tr.nodes_lonlat[a]]  # noqa: E731
    ys, xs = tr.lonlat[:, 1], tr.lonlat[:, 0]
    m = folium.Map(location=[ys.mean(), xs.mean()], zoom_start=14, tiles="OpenStreetMap")
    net = folium.FeatureGroup(f"road network ({len(ids)} arcs)")
    folium.PolyLine([ll(tr.arcs[i]) for i in ids], color="#888888", weight=1, opacity=0.7).add_to(net)
    net.add_to(m)
    rt = folium.FeatureGroup("reference route")
    folium.PolyLine([ll(tr.arcs[i]) for i in tr.route], color="#2ca02c", weight=4, opacity=0.8).add_to(rt)
    rt.add_to(m)
    sel = np.arange(0, len(ys), max(1, len(ys) // max_pts))
    g = folium.FeatureGroup("GPS points")
    for i in sel:
        folium.CircleMarker([ys[i], xs[i]], radius=2, color="#1f77b4", fill=True, fill_opacity=0.8, weight=1).add_to(g)
    g.add_to(m)
    lg = folium.FeatureGroup("labelled-arc points (colour = label distance)")
    for i in sel[lab[sel] >= 0]:
        c = "#2ca02c" if ld[i] < 5 else "#f1c40f" if ld[i] < 15 else "#e67e22" if ld[i] < 30 else "#d62728"
        folium.CircleMarker([ys[i], xs[i]], radius=4, color=c, fill=True, fill_opacity=0.9, weight=1,
                            tooltip=f"pt {i}, label dist {ld[i]:.1f} m").add_to(lg)
    lg.add_to(m)
    folium.LayerControl().add_to(m)
    m.fit_bounds([[ys.min(), xs.min()], [ys.max(), xs.max()]])
    legend = (f'<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;'
              f'border:1px solid #888;font:12px sans-serif"><b>{split} trip {tid}</b><br>'
              '<span style="color:#1f77b4">&#9679;</span> GPS point<br>'
              '<span style="color:#888">&#9473;</span> road network (bbox + 200 m)<br>'
              '<span style="color:#2ca02c">&#9473;</span> reference route<br>'
              'labelled point, distance to labelled arc:<br>'
              '<span style="color:#2ca02c">&#9679;</span> &lt;5 m '
              '<span style="color:#f1c40f">&#9679;</span> &lt;15 m '
              '<span style="color:#e67e22">&#9679;</span> &lt;30 m '
              '<span style="color:#d62728">&#9679;</span> 30-50 m<br>'
              f'<i>sampled: {len(sel)} of {len(ys)} points</i></div>')
    m.get_root().add_child(be.Element(legend))
    MAPS.mkdir(parents=True, exist_ok=True)
    out = MAPS / f"eda_{tid}.html"
    m.save(out)
    return out.stat().st_size / 1e6


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    rows, arr = [], {}
    for s in SPLITS:
        for t in list_trips(s):
            r, a = trip_row(s, t)
            rows.append(r)
            arr[s, t] = a
        print(f"{s}: done {time.time() - t0:.0f}s", flush=True)
    inv = pd.DataFrame(rows)
    inv.drop(columns=["lon0", "lat0"]).to_csv(RES / "eda_inventory.csv", index=False, float_format="%.5g")
    cand = {s: load_dataset(s) for s in SPLITS}
    summ = summarise(inv, cand)
    summ.to_csv(RES / "eda_summary.csv", index=False, float_format="%.5g")
    plot_all(inv, arr, cand)
    clean = inv[(inv.split == "train") & ~inv.trip_id.isin(EXCLUDED_TRIPS)]
    typ = clean.iloc[(clean.n_points - clean.n_points.median()).abs().argsort().iloc[0]].trip_id
    st = inv[inv.split == "student_test"]
    stt = st.iloc[(st.n_points - st.n_points.median()).abs().argsort().iloc[0]].trip_id
    traj_plot("train", typ, "traj_typical", " (typical)")
    traj_plot("train", BAD, "traj_bad_gt", " (bad ground truth)")
    for s, t in [("train", typ), ("student_test", stt)]:
        print(f"map {s} {t}: {folium_map(s, t):.2f} MB")
    print(f"typical={typ} student={stt} runtime={time.time() - t0:.0f}s")
    pd.set_option("display.width", 250, "display.max_columns", 99)
    print(summ.set_index("partition").T.to_string())


if __name__ == "__main__":
    main()

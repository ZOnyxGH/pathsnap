"""HMM/Viterbi map matching, route reconstruction and route metrics. Owner: B."""
import numpy as np
import pandas as pd

from .data import Trip


def viterbi(trip: Trip, cands: pd.DataFrame, p: np.ndarray, beta=5.0, lam=1.0, sigma=6.7) -> tuple[np.ndarray, list[int]]:
    """Return pred_arc (n,) int64 (-1 = no candidates) and the break indices."""
    raise NotImplementedError


def reconstruct_route(trip: Trip, pred_arc: np.ndarray) -> list[int]:
    """Full arc sequence from per-point arcs, with shortest-path gap filling."""
    raise NotImplementedError


def route_metrics(trip: Trip, pred_route: list[int], ref_route: np.ndarray) -> dict:
    """Keys: route_precision, route_recall, route_f1 (length-weighted), continuity, n_breaks."""
    raise NotImplementedError

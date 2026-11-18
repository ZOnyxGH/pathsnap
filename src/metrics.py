"""Point-level metrics shared by training and HMM evaluation. Owner: A."""
import numpy as np
import pandas as pd


def point_metrics(cands: pd.DataFrame, p: np.ndarray) -> dict:
    """Keys: top1, macro_f1, log_loss, miss_rate, n_points (PLAN section 5 definitions)."""
    raise NotImplementedError


def point_top1_from_assignment(label_arc: np.ndarray, pred_arc: np.ndarray) -> float:
    """Top-1 of a per-point arc assignment, ignoring points with label_arc == -1."""
    raise NotImplementedError

"""Six-model comparison, GroupKFold CV, plots and the final model bundle. Owner: A."""
import numpy as np
import pandas as pd


def predict_proba(bundle: dict, cands: pd.DataFrame) -> np.ndarray:
    """P(label=1) per candidate row, using bundle["features"] and bundle["model"]."""
    raise NotImplementedError


def main() -> None:
    """CLI entry: python -m src.train."""
    raise NotImplementedError

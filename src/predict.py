"""End-to-end single-trip inference, used by evaluate.py and the API. Owner: B."""


def predict_trip(split: str, trip_id: str, bundle: dict, use_hmm: bool = True) -> dict:
    """Return {"trip_id", "pred_arc", "route", "route_lonlat", "timing_ms"}; never reads .route."""
    raise NotImplementedError

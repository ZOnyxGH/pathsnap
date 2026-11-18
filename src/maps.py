"""Folium maps of GPS track and predicted route. Owner: B."""
from .data import Trip


def make_map(trip: Trip, route: list[int]):
    """Return a folium.Map of the GPS points and the predicted route (no reference layer)."""
    raise NotImplementedError

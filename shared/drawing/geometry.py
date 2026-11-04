"""Geometry type aliases: Cell and Wall (maze), Point (path_tracing)."""
from __future__ import annotations

Cell = tuple[int, int]          # (col, row) integer grid coordinate
Wall = tuple[Cell, Cell]        # an undirected wall between two adjacent cells
Point = tuple[float, float]     # a pixel-space point

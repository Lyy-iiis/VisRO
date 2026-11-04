"""Robust separation of line ink from dot ink.

Idea: look at the *ink vector* d = 255 - pixel.
  * line ink has d parallel to (255 - line_rgb), a fixed direction;
  * black-dot ink has d parallel to the grey axis (1,1,1).
Split by direction (cosine), not by brightness -- an antialiased,
semi-transparent line still points the same way, so this does not
misfire on light line colours (pink / purple) the way a
"channel spread" threshold does.
"""
import numpy as np


def masks(A, col):
    """Return (line_mask, dot_mask) boolean arrays for an HxWx3 int array."""
    d = 255.0 - A
    n = np.linalg.norm(d, axis=2) + 1e-6
    vl = 255.0 - np.asarray(col, float); vl /= np.linalg.norm(vl)
    vg = np.array([1., 1., 1.]) / np.sqrt(3)
    cl = (d @ vl) / n
    cg = (d @ vg) / n
    line = (n > 25) & (cl > 0.985) & (cl > cg)
    dot = (n > 60) & (cg > 0.97)
    return line, dot

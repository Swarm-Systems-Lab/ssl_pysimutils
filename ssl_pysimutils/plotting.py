"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os

# Algebra
import numpy as np
from numpy import linalg as la

# External graphic tools
import matplotlib.pylab as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# -----------------------------------------------------------------------------
# Plotting tools
# -----------------------------------------------------------------------------


def unicycle_patch(XY, yaw, color, size=1, lw=0.5):
    """
    Unicycle patch.
    * XY: position [X, Y] of the patch
    * yaw: heading of the unicycle
    """
    Rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])

    apex = 45 * np.pi / 180  # 30 degrees apex angle
    b = np.sqrt(1) / np.sin(apex)
    a = b * np.sin(apex / 2)
    h = b * np.cos(apex / 2)

    z1 = size * np.array([a / 2, -h * 0.3])
    z2 = size * np.array([-a / 2, -h * 0.3])
    z3 = size * np.array([0, h * 0.6])

    z1 = Rot.dot(z1)
    z2 = Rot.dot(z2)
    z3 = Rot.dot(z3)

    verts = [
        (XY[0] + z1[1], XY[1] + z1[0]),
        (XY[0] + z2[1], XY[1] + z2[0]),
        (XY[0] + z3[1], XY[1] + z3[0]),
        (0, 0),
    ]

    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    path = Path(verts, codes)

    return patches.PathPatch(path, fc=color, lw=lw)


def zoom_range(begin, end, center, scale_factor):
    """
    Compute a 1D range zoomed around center.
    (moded from https://gist.github.com/dukelec/e8d4171ef4d12f9998295cfcbe3027ce)
    * begin: The begin bound of the range.
    * end: The end bound of the range.
    * center: The center of the zoom (i.e., invariant point)
    * scale_factor: The scale factor to apply.
    :return: The zoomed range (min, max)
    """
    if begin < end:
        min_, max_ = begin, end
    else:
        min_, max_ = end, begin

    old_min, old_max = min_, max_

    offset = (center - old_min) / (old_max - old_min)
    range_ = (old_max - old_min) / scale_factor
    new_min = center - offset * range_
    new_max = center + (1.0 - offset) * range_

    if begin < end:
        return new_min, new_max
    else:
        return new_max, new_min


def alpha_cmap(cmap, alpha):
    """
    Apply alpha to the desired color map
    https://stackoverflow.com/questions/37327308/add-alpha-to-an-existing-matplotlib-colormap
    ----------------------------------------------------------------------

    When using pcolormesh, directly applying alpha can cause many problems.
    The ideal approach is to generate and use a pre-diluted color map on a white background.
    """
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Define the alphas in the range from 0 to 1
    alphas = np.linspace(alpha, alpha, cmap.N)

    # Define the background as white
    BG = np.asarray(
        [
            1.0,
            1.0,
            1.0,
        ]
    )

    # Mix the colors with the background
    for i in range(cmap.N):
        my_cmap[i, :-1] = my_cmap[i, :-1] * alphas[i] + BG * (1.0 - alphas[i])

    # Create new colormap which mimics the alpha values
    my_cmap = ListedColormap(my_cmap)
    return my_cmap

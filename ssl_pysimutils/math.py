"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os

# Algebra
import numpy as np
from numpy import linalg as la

# -----------------------------------------------------------------------------
# Math tools
# -----------------------------------------------------------------------------


def pprz_angle(theta_array: np.ndarray):
    return -theta_array + np.pi / 2

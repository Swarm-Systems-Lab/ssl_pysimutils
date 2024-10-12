"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import pandas as pd

# Algebra
import numpy as np
from numpy import linalg as la

# -----------------------------------------------------------------------------
# Common tools (general use utility functions)
# -----------------------------------------------------------------------------


def load_data(filename, t0, tf=None, sep="\t", time_label="Time"):
    """
    Load Paparazzi .csv data files
    """
    data = pd.read_csv(filename, sep=sep)
    if tf is None:
        data = data.loc[(data[time_label] >= t0)]
    else:
        data = data.loc[(data[time_label] >= t0) & (data[time_label] <= tf)]
    return data


def createDir(dir, verbose=True):
    """Create a new directory if it doesn't exist"""
    try:
        os.mkdir(dir)
        if verbose:
            print("Directory '{}' created!".format(dir))
    except FileExistsError:
        if verbose:
            print("The directory '{}' already exists!".format(dir))


# """
# Check if the dimensions are correct and adapt the input to 2D.
# """
# def two_dim(X):
#     if type(X) == list:
#         return np.array([[X]])
#     elif len(X.shape) < 2:
#         return np.array([X])
#     else:
#         return X

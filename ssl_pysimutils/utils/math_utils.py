"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os

# Algebra
import numpy as np
from numpy import linalg as la

import random

# --------------------------------------------------------------------------------------
# Math tools
# --------------------------------------------------------------------------------------


def unit_vec(v):
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return v


def R_2D_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def pprz_angle(theta_array: np.ndarray):
    return -theta_array + np.pi / 2


# --------------------------------------------------------------------------------------
# Statistical distributions


def uniform_distrib(N: int, lims: list[float], rc0: list[float] = [0, 0], seed=None):
    """
    Function to generate uniform rectangular distributions

    Attributes
    ----------
    N:
        number of points
    rc0:
        position [x,y,...] in the real space of the centroid
    lims:
        distance limits [lim_x,lim_y,...] in each dimension of the real space
    """
    if seed is not None:
        random.seed(seed)

    if len(rc0) + len(lims) != 2 * 2:
        raise Exception("The dimension of rc0 and lims should be 2")

    X0 = (np.random.rand(N, 2) - 0.5) * 2
    for i in range(2):
        X0[:, i] = X0[:, i] * lims[i]
    return rc0 + X0


# -------------------------------------------------------------------------------------
# Laplacian utils


def build_B(list_edges, N):
    """
    Generate the incidence matrix
    """
    B = np.zeros((N, len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0], i] = 1
        B[list_edges[i][1], i] = -1
    return B


def build_L_from_B(B):
    """
    Generate the Laplacian matrix by using the incidence matrix (unit weights)
    """
    L = B @ B.T / 2
    return L


# -------------------------------------------------------------------------------------
# Graph generators


def gen_Z_random(N: int, rounds: int = 1, seed=None):
    """
    Generate a random graph using a simple heuristic
    """
    if seed is not None:
        random.seed(seed)

    Z = []

    while rounds:
        non_visited_nd = set(range(N))
        non_visited_nd.remove(0)
        visited_nd = {0}

        while len(non_visited_nd) != 0:
            i = random.choice(list(visited_nd))
            j = random.choice(list(non_visited_nd))
            visited_nd.add(j)
            non_visited_nd.remove(j)

            if (i, j) not in Z:
                Z.append((i, j))

        rounds -= 1

    return Z


def gen_Z_distance(P: np.ndarray, dist_thr: float):
    """
    Generate a graph using a distance base heuristic:
        If d_ij <= dist_thr then append (i,j) to Z
    """
    y2 = np.sum(P**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * P @ P.T + y2)

    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr
    Z = [(i, j) for i, j in zip(*np.where(mask))]
    return Z


def gen_Z_split(N: int, order: int, n_breaks: int = 0):
    """
    Split a full conected graph into "n_breaks" full connected graphs.
    Then, remove N/order connections
    """
    X = np.ones((N, 2))
    for i in range(order):
        if i != order - 1:
            X[i * int(N / order) : (i + 1) * int(N / order), :] = [i, i]
        else:
            X[i * int(N / order) :, :] = [i, i]

    y2 = np.sum(X**2, axis=1)
    x2 = y2.reshape(-1, 1)
    dist = np.sqrt(x2 - 2 * X @ X.T + y2)

    dist_thr = 0.1
    mask = dist + 2 * np.eye(dist.shape[0]) * dist_thr <= dist_thr

    Z = [(i, j) for i, j in zip(*np.where(mask))]

    # Remove some conections
    N_subgraph = int(N / order)
    edges_subgraph = int(2 * N_subgraph * (N_subgraph - 1) / 2)
    idx_to_remove = []
    if n_breaks > 0:
        for i in range(order):
            for j in range(n_breaks):
                idx = edges_subgraph * i + int(N * N_subgraph / n_breaks) * j
                idx_to_remove.append(idx)
    Z = [edge for i, edge in enumerate(Z) if i not in idx_to_remove]
    return Z


def gen_Z_ring(N: int):
    """
    Generate a ring graph
    """
    Z = [(i, i + 1) for i in range(N - 1)]
    Z.extend([(i + 1, i) for i in range(N - 1)])
    Z.append((N - 1, 0))
    Z.append((0, N - 1))
    return Z


# -------------------------------------------------------------------------------------

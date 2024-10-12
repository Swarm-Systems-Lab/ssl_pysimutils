"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA

# ------------------------------------------------------------------------------------
# Debugging tools
# ------------------------------------------------------------------------------------


def debug_eig(A, eigenvectors=True, prec_values=8, prec_vectors=3):
    eg = LA.eig(A)

    with np.printoptions(precision=prec_values, suppress=True):
        print(" --- Eigenvalues")
        for i in range(len(eg[0])):
            print("lambda_{:d} = {:f}".format(i, eg[0][i]))

    if eigenvectors:
        with np.printoptions(precision=prec_vectors, suppress=True):
            print("--- Eigenvectors")
            for i in range(len(eg[0])):
                print("v_{:d} =".format(i), eg[1][:, i])


# ------------------------------------------------------------------------------------

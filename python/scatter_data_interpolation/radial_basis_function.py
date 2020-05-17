from math import log
from math import exp
from math import sqrt

import numpy
# from scipy.sparse import lil_matrix
from scipy.spatial.distance import euclidean

# ==============================================================================
#                                KERNELS
# ==============================================================================


def linear(r, c):
    return 1.0-r/c if r < c else 0.0


def gaussian(r, c):
    return exp(-(r**2/(2*c**2)))


def inverse_multiquadric(r, c):
    return 1.0/sqrt(r**2 + c**2)


def thin_plate_spline(r, *args):
    r = max(r, 1e-8)
    return r**2 * log(r)


def multiquadric(r, c):
    return sqrt(1.0+(c*r)**2)/2.0


def biharmonic(r, *args):
    return r


def hardy_multiquadric(r, c):
    return sqrt(r**2 + c**2)


# ==============================================================================
#                                SOLVER
# ==============================================================================

def solve_weights(nodes, values, sigma=1.0,
                  distance_function=euclidean, kernel_function=linear):

    n = len(nodes)
    A = numpy.eye(n)

    for i in range(n):
        for j in range(n):
            A[i, j] = kernel_function(distance_function(nodes[i], nodes[j]), sigma)

    # solve weights
    return numpy.linalg.solve(A, values)


def rbf(current_node, nodes, weights, sigma=1.0,
        distance_function=euclidean, kernel_function=linear):

    n = len(nodes)
    a = numpy.array([kernel_function(distance_function(current_node, nodes[i]), sigma) for i in range(n)])
    return numpy.dot(a, weights)

    # A = lil_matrix((n, n))
    # A.setdiag([kernel_function(distance_function(current_node, nodes[i]), sigma) for i in range(n)])
    # return numpy.sum(A*weights, axis=0)

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


def exponential(r, c):
    return exp(-r * (1.0/c**2))


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


KERNELS = [
    linear,
    exponential,
    gaussian,
    inverse_multiquadric,
    thin_plate_spline,
    multiquadric,
    biharmonic,
    hardy_multiquadric
]

# ==============================================================================
#                                SOLVER
# ==============================================================================

def distance_matrix(nodes, distance_function=euclidean):

    n = len(nodes)
    A = numpy.eye(n)

    for i in range(n):
        for j in range(i, n):
            A[i, j] = A[j, i] = distance_function(nodes[i], nodes[j])

    return A

def average_sigma(nodes, weight=0.5):
    '''
    Computes averaged sigma based on closest and farthest nodes
    '''

    def upper_tri_indexing(A):
        '''Returns upper triangle without diagonals'''
        m = A.shape[0]
        r,c = numpy.triu_indices(m, 1)
        return A[r,c]
    
    weight = min(max(weight, 0.0), 1.0)
    D = distance_matrix(nodes)
    upper_triangle = upper_tri_indexing(D) # D[numpy.triu_indices_from(D)]

    _min = numpy.min(upper_triangle)
    _max = numpy.max(upper_triangle)
    
    return ((1-weight) * _min) + (weight * _max)

def solve_weights(nodes, values, sigma=1.0, 
                  distance_function=euclidean, kernel_function=linear):

    # build A
    D = distance_matrix(nodes, distance_function=distance_function)
    A = numpy.vectorize(kernel_function)(D, sigma)
    b = values

    # Ax = b
    # (A^-1)Ax = (A^-1)b
    # x = (A^-1)b
    Ai = numpy.linalg.inv(A)
    w = Ai.dot(b)

    # solve weights
    # w = numpy.linalg.solve(A, values)

    return w

def rbf(current_node, nodes, weights, sigma=1.0,
        distance_function=euclidean, kernel_function=linear):

    n = len(nodes)
    a = numpy.array([kernel_function(distance_function(current_node, nodes[i]), sigma) for i in range(n)])
    
    return numpy.dot(a, weights)

    # A = lil_matrix((n, n))
    # A.setdiag([kernel_function(distance_function(current_node, nodes[i]), sigma) for i in range(n)])
    # return numpy.sum(A*weights, axis=0)

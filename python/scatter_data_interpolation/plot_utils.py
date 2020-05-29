import numpy
from matplotlib import pyplot

from .radial_basis_function import rbf
from .radial_basis_function import solve_weights


def plot_kernels(kernels):
    '''
    Plots Radial Basis Function Kernels.
    '''
    n = len(kernels)

    # range
    x = numpy.linspace(-5, 5, 100)

    # create plot figures
    f = pyplot.figure(figsize=(30, 30))
    for i, (name, func, c) in enumerate(kernels):

        ax = f.add_subplot(n, n, i+1)
        ax.title.set_text(name)

        y = numpy.vectorize(func)(abs(x), c)
        ax.plot(x, y)


def plot_1d(kernels, nodes, values):
    '''
    Plots 1D RBF
    '''

    n = len(kernels)
    x = numpy.linspace(0, 1, 100)[:, numpy.newaxis]

    f = pyplot.figure(figsize=(30, 30))
    for i, (name, kernel_func, c) in enumerate(kernels):

        ax = f.add_subplot(n, n, i+1)
        ax.title.set_text(name)
        if c is not None:
            ax.set_xlabel('Sigma: {}'.format(c))

        # solve weights
        weights = solve_weights(nodes, values, kernel_function=kernel_func, sigma=c)

        # run rbf function
        def func(x): return rbf(x, nodes, weights, kernel_function=kernel_func, sigma=c)
        y = numpy.vectorize(func)(x)

        ax.plot(x, y)
        ax.scatter(nodes, values, s=100, c='red')
        ax.set_ybound(lower=0.0, upper=1)


def plot_2d(kernels, nodes, values):
    '''
    Plots 2D RBF
    '''

    # create NxN grid
    N = 100
    ti = numpy.linspace(0, 1, N)

    # create plot figures
    n = len(kernels)
    f = pyplot.figure(figsize=(30, 30))
    for i, (name, kernel_func, c) in enumerate(kernels):

        ax = f.add_subplot(n, n, i+1)
        ax.title.set_text(name)
        if c is not None:
            ax.set_xlabel('Sigma: {}'.format(c))

        weights = solve_weights(nodes, values, kernel_function=kernel_func, sigma=c)
        data = numpy.array([rbf([ti[j], ti[i]], nodes, weights, kernel_function=kernel_func, sigma=c) for i in range(N) for j in range(N)])

        ax.scatter(nodes[:, 0]*N, nodes[:, 1]*N, s=50, c=values, edgecolor='black')
        ax.imshow(data.reshape(N, N, 3), origin='lower')

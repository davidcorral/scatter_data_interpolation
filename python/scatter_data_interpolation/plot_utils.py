'''
Refs:
- https://scipython.com/book/chapter-7-matplotlib/examples/simple-surface-plots/

'''
import numpy
from matplotlib import cm
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

from .radial_basis_function import rbf
from .radial_basis_function import average_sigma
from .radial_basis_function import solve_weights

def pretty_name(snake_case):
    return ' '.join([token.capitalize() for token in snake_case.split('_')])

def plot_kernels(kernels, c):
    '''
    Plots Radial Basis Function Kernels.
    '''
    n = len(kernels)

    # range
    x = numpy.linspace(-5, 5, 100)

    # create plot figures
    f = pyplot.figure(figsize=(30, 30))
    for i, kernel_func in enumerate(kernels):
        
        ax = f.add_subplot(n, n, i+1)
        ax.title.set_text(pretty_name(kernel_func.__name__))

        y = numpy.vectorize(kernel_func)(abs(x), c)
        ax.plot(x, y)


def plot_1d(kernels, nodes, values, figsize=(30, 30)):
    '''
    Plots 1D RBF
    '''

    n = len(kernels)
    x = numpy.linspace(0, 1, 100)[:, numpy.newaxis]

    c = average_sigma(nodes, 0.75)

    f = pyplot.figure(figsize=figsize)
    for k, kernel_func in enumerate(kernels):

        ax = f.add_subplot(n, n, k+1)
        ax.title.set_text(pretty_name(kernel_func.__name__))
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

def plot_2d(kernels, nodes, values, sigma=None, figsize=(30, 30)):
    '''
    Plots 2D RBF
    '''

    # create NxN grid
    N = 100
    ti = numpy.linspace(0, 1, N)

    if sigma is None:
        sigma = average_sigma(nodes, 0.75)

    # create plot figures
    n = len(kernels)
    f = pyplot.figure(figsize=figsize)
    for k, kernel_func in enumerate(kernels):

        ax = f.add_subplot(n, n, k+1)
        ax.title.set_text(pretty_name(kernel_func.__name__))
        ax.set_xlabel('Sigma: {}'.format(sigma))

        weights = solve_weights(nodes, values, kernel_function=kernel_func, sigma=sigma)
        data = numpy.array([rbf([ti[j], ti[i]], nodes, weights, kernel_function=kernel_func, sigma=sigma) for i in range(N) for j in range(N)])

        ax.scatter(nodes[:, 0]*N, nodes[:, 1]*N, s=50, c=values, edgecolor='black')
        ax.imshow(data.reshape(N, N, 3), origin='lower')


def plot_3d(kernels, nodes, values, sigma=None, figsize=(30, 30)):
    
    # Create grid
    N = 100
    x = numpy.linspace(0, 1, N)
    y = x.copy()
    X, Y = numpy.meshgrid(x, y)

    if sigma is None:
        sigma = average_sigma(nodes, 0.75)

    # create plot figures
    n = len(kernels)
    f = pyplot.figure(figsize=figsize)
    for k, kernel_func in enumerate(kernels):
        
        # ax = f.add_subplot(projection='3d')
        # ax = f.gca(projection='3d')
        ax = f.add_subplot(n, n, k+1, projection='3d') 
        ax.title.set_text(pretty_name(kernel_func.__name__))
        ax.set_xlabel('Sigma: {}'.format(sigma))

        weights = solve_weights(nodes, values, kernel_function=kernel_func, sigma=sigma)
        data = numpy.array([rbf([x[j], y[i]], nodes, weights, kernel_function=kernel_func, sigma=sigma) for i in range(N) for j in range(N)])

        # Plot the surface.
        # rstride=12, cstride=12, 
        # c = cm.jet((values/numpy.amax(values)).ravel())
        # surf = ax.plot_surface(X, Y, data.reshape(N, N), #cmap='jet', 
        #                     linewidth=1, edgecolor='black', antialiased=True, zorder=0, alpha=0)
        ax.plot_wireframe(X, Y, data.reshape(N, N),
                          linewidth=1, edgecolor='black', antialiased=True, zorder=0)

        # Plot sample points
        ax.scatter(nodes[:, 0], nodes[:, 1], values, s=50, edgecolor='black', c='red', zorder=1)

        # Customize the z axis.
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

    pyplot.show()
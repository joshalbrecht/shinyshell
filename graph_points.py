import numpy
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j

    m, residuals, _, singular = numpy.linalg.lstsq(G, z)
    print "coeffs", m
    print "residuals", residuals
    print "singular values of matrix", singular
    return m

def main():
    points = []
    f = open('pts.points', 'r')
    for line in f:
        point = line.strip('[]\n').split()
        point = numpy.array(point, dtype='|S4')
        point = point.astype(numpy.float)
        points.append(point)

    points = numpy.vstack(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    coefficients = polyfit2d(x, y, z, order=3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

if __name__ == '__main__':
    main()

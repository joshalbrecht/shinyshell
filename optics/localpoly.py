
import itertools

import numpy
import matplotlib.pyplot

from optics.base import * # pylint: disable=W0401,W0614
import optics.taylor_poly
import optics.debug

def fit_poly(space, points, screen_point, radius, poly_order):
    """
    Fits a taylor polynomial to the points and creates a LocalTaylorPoly
    """
    #fit the polynomial to the points:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    coefficients, surface_is_retarded = _polyfit2d(x, y, z, poly_order)
    assert not surface_is_retarded, "Very poorly fit TaylorPoly :("
    order = int(numpy.sqrt(len(coefficients)))
    cohef = []
    for i in range(0, order):
        cohef.append(coefficients[i*order:(i+1)*order])
    cohef = numpy.array(cohef).copy(order='C')
    fudge = 0.001
    poly = optics.taylor_poly.TaylorPoly(cohef=cohef.T, min_z=numpy.min(z)-fudge, max_z=numpy.max(z)+fudge, domain_radius=radius, domain_point=screen_point)
    return LocalTaylorPoly(poly, space)

def _polyfit2d(x, y, z, order):
    """
    Figure out the coefficients that best fit (least squares) a polynomial of order to the x/y/z data
    """
    
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    indices = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(indices):
        G[:, k] = x**i * y**j
    coefficients, residuals, _, _ = numpy.linalg.lstsq(G, z)
    residual_sum = sum(residuals)
    print("Taylor poly fit: %s" % (residual_sum))
    return coefficients, residual_sum > 1.0

class LocalTaylorPoly(object):
    """
    Hides the transformation of the taylor poly so that we can intersect with it in a normal way
    """
    
    def __init__(self, poly, space):
        self.poly = poly
        self.space = space
        
    def get_z_for_plot(self, x, y):
        return self.poly.eval_poly(x, y)
        
    def _local_reflect_rays(self, rays):
        """
        :returns: all of the rays, reflected off of this surface (or None if the ray did not hit the surface)
        """
        reflected_rays = []
        reflection = None
        for ray in rays:
            intersection, normal = self._intersection_plus_normal(ray.start, ray.end)
            if intersection == None:
                reflected_rays.append(None)
            else:
                reverse_ray_direction = normalize(ray.start - ray.end)
                midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
                reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
                reflection = intersection + reflection_direction
                reflected_rays.append(Ray(intersection, reflection))
            if optics.debug.TAYLOR_SURFACE_REFLECTIONS:
                #TODO: graph the rays and reflection and normal if they exist
                axes = matplotlib.pyplot.subplot(111, projection='3d')
                size = 5
                num_points = 10
                x, y = numpy.meshgrid(numpy.linspace(-size, size, num_points), numpy.linspace(-size, size, num_points))
                axes.scatter(x, y, self.get_z_for_plot(x, y), c='r', marker='o').set_label('grid')
                axes.plot([ray.start[0], ray.end[0]], [ray.start[1], ray.end[1]], [ray.start[2], ray.end[2]], label="ray")
                axes.set_xlabel('X')
                axes.set_ylabel('Y')
                axes.set_zlabel('Z')
                matplotlib.pyplot.legend()
                print reflected_rays
                matplotlib.pyplot.show()
        return reflected_rays
        
    def _intersection_plus_normal(self, start, end):
        """
        Just like intersection, but returns the normal as well
        
        Assumes points are in space coordinates already
        """
        point = self.poly._intersection(start, normalize(end-start))
        if point == None:
            return None, None
        #calculate the normal as well
        normal = self.poly.normal(point)
        return point, normal
    

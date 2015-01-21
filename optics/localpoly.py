
import itertools

from optics.base import * # pylint: disable=W0401,W0614
import optics.taylor_poly

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
    return coefficients, sum(residuals) > 1.0

class LocalTaylorPoly(object):
    """
    Hides the transformation of the taylor poly so that we can intersect with it in a normal way
    """
    
    def __init__(self, poly, space):
        self.poly = poly
        self.space = space
        
    def get_z_for_plot(self, x, y):
        return self.poly.eval_poly(x, y)
        
    def reflect_rays(self, rays):
        """
        :returns: all of the rays, reflected off of this surface (or None if the ray did not hit the surface)
        """
        reflected_rays = []
        for ray in rays:
            projected_start = self.space.point_to_space(ray.start)
            projected_end = self.space.point_to_space(ray.end)
            intersection, normal = self._intersection_plus_normal(projected_start, projected_end)
            if intersection == None:
                reflected_rays.append(None)
            else:
                reflected_rays.append(Ray(self.space.point_from_space(intersection), self.space.point_from_space(intersection+normal)))
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
    

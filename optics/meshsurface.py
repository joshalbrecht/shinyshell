
"""
Just used by main.py to allow triangle mesh surfaces to be used in pyoptools
"""

import vtk

import numpy
from numpy import  array, asarray, arange, polyadd, polymul, polysub, polyval,\
     dot, inf, roots, zeros, meshgrid, sqrt,where, abs,  isreal

from pyoptools.raytrace.surface.surface import Surface
from pyoptools.raytrace.ray.ray import Ray

import mesh

class MeshSurface(Surface):

    def __init__(self, arcs, *args, **kwargs):
        Surface.__init__(self, *args, **kwargs)
        self._mesh = mesh.Mesh(mesh.mesh_from_arcs(arcs))
        self._inf_vector = numpy.array((inf,inf,inf))

    def topo(self, xarray, yarray):
        """**Returns the Z value for a given X and Y**

        This method returns the topography of the polynomical surface to be
        used to plot the surface.
        """
        result = zeros(xarray.shape)
        for i in range(0, len(xarray)):
            point = self._get_point_and_normal(xarray[i], yarray[i])[0]
            if point == None:
                result[i] = 0
            else:
                result[i] = point[2]
        return result

    def _get_point_and_normal(self, x, y):
        pointRaySource = (x, y, -1000000.0)
        pointRayTarget = (x, y, 1000000.0)
        return self._mesh.intersection_plus_normal(pointRaySource, pointRayTarget)

    def _intersection(self, A):
        '''**Point of intersection between a ray and the polynomical surface**

        This method returns the point of intersection  between the surface
        and the ray. This intersection point is calculated in the coordinate
        system of the surface.

           iray -- incident ray

        iray must be in the coordinate system of the surface
        '''

        #TODO: should be way smarter than this and use normals to sort things out (don't hit when coming from reverse side of surface)
        #basically the problem is that we're currently colliding right at the start of the ray
        pointRaySource = A.pos + A.dir * 0.001
        pointRayTarget = (1000000.0 * numpy.array(A.dir)) + numpy.array(A.pos)
        
        point, normal = self._mesh.intersection_plus_normal(pointRaySource, pointRayTarget)
        if point == None:
            return self._inf_vector
        return numpy.array(point)


    def normal(self, int_p):
        """**Return the vector normal to the surface**

        This method returns the vector normal to the polynomical surface at a
        point ``int_p=(x,y,z)``.

        Note: It uses ``x`` and ``y`` to calculate the ``z`` value and the normal.
        """

        normal = self._get_point_and_normal(int_p[0], int_p[1])[1]
        if normal == None:
            return numpy.array((0,0,1))
        return normal


    def _repr_(self):
        '''
        Return an string with the representation of the mesh surface
        '''
        return "MeshSurface(mesh="+str(self._mesh)+")"


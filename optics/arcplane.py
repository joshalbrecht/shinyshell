
from optics.base import *
import optics.globals
import optics.rotation_matrix

class ArcPlane(object):
    """
    Immutable. Please do not try to change rho or mu after creating the plane.
    
    mu=rotation about the Y axis (vertical planes)
    rho=rotation about the X axis (horizontal planes)
    
    Plane coordinates are as follows--the center is still at the eye at 0,0,0,
    but positive x is towards the shell, and positive y is towards the screen
    """
    
    def __init__(self, rho=None, mu=None):
        assert (rho == None) != (mu == None), "Exactly one of rho or mu must be None"
        self.rho = rho
        self.mu = mu
        
        if self.rho == None:
            self.angle = self.mu
            self.rotation_axis = Point3D(0.0, 1.0, 0.0)
            base_plane_normal = Point3D(1.0, 0.0, 0.0)
        else:
            self.angle = self.rho
            self.rotation_axis = Point3D(1.0, 0.0, 0.0)
            base_plane_normal = Point3D(0.0, 1.0, 0.0)
            
        self.local_to_world_rotation_matrix = numpy.zeros((3,3))
        optics.rotation_matrix.R_axis_angle(self.local_to_world_rotation_matrix, self.rotation_axis, self.angle)
        self.world_to_local_rotation_matrix = numpy.linalg.inv(self.local_to_world_rotation_matrix)
        base_view_normal = Point3D(0.0, 0.0, -1.0)
        self.view_normal = self.local_to_world_rotation_matrix.dot(base_view_normal)
        self.normal = self.local_to_world_rotation_matrix.dot(base_view_normal)
        self.plane = Plane(Point3D(0.0, 0.0, 0.0), self.normal)
        
    def world_to_local(self, point):
        """
        Convert a Point3D from world space into a Point2D in plane space.
        Note: make no guarantees that this is a good idea. In particular,
        be careful not to call this with things that should not be projected
        on to the plane.
        """
        flat_point = self.world_to_local_rotation_matrix.dot(point)
        if self.rho == None:
            return Point2D(-flat_point[2], flat_point[1])
        else:
            return Point2D(flat_point[0], -flat_point[2])
    
    def local_to_world(self, point):
        """
        Converts from a Point2D in plane space into a Point3D in world space.
        Is always reasonable to call.
        """
        if self.rho == None:
            flat_point = Point3D(0.0, point[1], -point[0])
        else:
            flat_point = Point3D(point[0], 0.0, -point[1])
        return self.local_to_world_rotation_matrix.dot(flat_point)
    
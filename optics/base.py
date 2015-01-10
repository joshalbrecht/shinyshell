
"""
Shared code between main.py and interface.py.
"""

import math
import collections

import numpy
import OpenGL.GL

import rotation_matrix

#these are very simple classes. just an alias for a numpy array that gives a little more intentionality to the code
def Point2D(*args):
    return numpy.array(args)

def Point3D(*args):
    return numpy.array(args)

#Used for vision rays. See module docstring
AngleVector = collections.namedtuple('AngleVector', ['theta', 'phi'])

class Plane(object):
    def __init__(self, point, normal):
        self._point = point
        self._normal = normal
        
    # intersection function
    def intersect_line(self, p0, p1, epsilon=0.0000001):
        """
        p0, p1: define the line    
        return a Vector or None (when the intersection can't be found).
        """
    
        u = p1 - p0
        w = p0 - self._point
        dot = self._normal.dot(u)
    
        if abs(dot) > epsilon:
            # the factor of the point between p0 -> p1 (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # otherwise:
            #  < 0.0: behind p0.
            #  > 1.0: infront of p1.
            fac = -self._normal.dot(w) / dot
            return p0 + u*fac
        else:
            # The segment is parallel to plane
            return None

class Ray(object):
    def __init__(self, start, end):
        self._start = start
        self._end = end
        
    @property
    def start(self): 
        return self._start
    
    @property
    def end(self): 
        return self._end
    
class VisibleLineSegment(Ray):
    def __init__(self, start, end, color=(1.0, 1.0, 1.0)):
        Ray.__init__(self, start, end)
        self._color = color
        
    def render(self):
        OpenGL.GL.glBegin(OpenGL.GL.GL_LINES)
        OpenGL.GL.glColor3f(*self._color)
        OpenGL.GL.glVertex3f(*self._start)
        OpenGL.GL.glVertex3f(*self._end)
        OpenGL.GL.glEnd()

class LightRay(VisibleLineSegment):
    def __init__(self, start, end):
        VisibleLineSegment.__init__(self, start, end, color=(0.5, 0.5, 0.5))

def get_arc_plane_normal(principal_ray, is_horizontal):
    """
    Defines the normal for the arc plane (the plane in which the arc will reside)
    If the principal ray is (0,0,-1) and:
        is_horizontal=True: the arc plane normal will be (0,1,0)
        is_horizontal=False: the arc plane normal will be (1,0,0)
    Otherwise, those vectors will undergo the same rotation as the principal ray
    """
    base_principal_ray = Point3D(0.0, 0.0, -1.0)
    ray_rotation = numpy.zeros((3, 3))
    rotation_matrix.R_2vect(ray_rotation, base_principal_ray, principal_ray)
    base_arc_ray = Point3D(1.0, 0.0, 0.0)
    if is_horizontal:
        base_arc_ray = Point3D(0.0, 1.0, 0.0)
    return ray_rotation.dot(base_arc_ray)

def normalize(a):
    return a / numpy.linalg.norm(a)

def normalized_vector_angle(v1_u, v2_u):
    """ Returns the angle in radians between normal vectors 'v1_u' and 'v2_u'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    angle = numpy.arccos(numpy.dot(v1_u, v2_u))
    if numpy.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return numpy.pi
    return angle

def _create_transform_matrix_from_rotations(rotations):
    """
    Note: do not use this in new code
    
    'rotations' is a really dumb representation / notion of rotation from pyOpTools
    Basically they want you to define a rotation as the tuple:
    
    (x_rot, y_rot, z_rot)
    
    Which defines a set of 3 rotations that are applied sequentially.
    ie, rotate around the x-axis by x_rot, THEN around the y axis by y_rot, then around
    the z axis by z_rot.
    
    This function converts that nonsense into the resulting rotation matrix
    """
    c = numpy.cos(rotations)
    s = numpy.sin(rotations)

    rx = numpy.array([[1. , 0., 0.],
    [0. , c[0],-s[0]],
    [0. , s[0], c[0]]])

    ry = numpy.array([[ c[1], 0., s[1]],
    [ 0., 1., 0.],
    [-s[1], 0., c[1]]])

    rz = numpy.array([[ c[2],-s[2], 0.],
    [ s[2], c[2], 0.],
    [ 0., 0., 1.]])

    return numpy.dot(rz, numpy.dot(ry, rx))

def angle_vector_to_vector(angle_vec, principal_ray):
    """
    Converts from a (phi, theta) pair to a normalized Point3D()
    """
    phi_rot = _create_transform_matrix_from_rotations((0,-angle_vec.phi,0))
    ray = phi_rot.dot(principal_ray)
    theta_rot = _create_transform_matrix_from_rotations((0,0,angle_vec.theta))
    return theta_rot.dot(ray)

def dist2(v, w):
    return sum(((math.pow(v[i] - w[i], 2) for i in range(0, len(v)))))

def distToLineSquared(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return dist2(p, v)
    n = w - v
    t = ((p - v).dot(n)) / l2
    return dist2(p, v + t * n)

def distToSegmentSquared(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return dist2(p, v)
    n = w - v
    t = ((p - v).dot(n)) / l2
    if (t < 0):
        return dist2(p, v)
    if (t > 1):
        return dist2(p, w)
    return dist2(p, v + t * n)

def closestPointOnLine(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return dist2(p, v)
    n = w - v
    t = ((p - v).dot(n)) / l2
    return v + t * n

def distToSegment(p, v, w):
    return math.sqrt(distToSegmentSquared(p, v, w))

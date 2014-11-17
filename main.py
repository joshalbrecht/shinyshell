#!/user/bin/python

import itertools
import math
from collections import namedtuple

import numpy
import numpy.linalg

from OpenGL.GL import *
from OpenGL.GLU import *

import wxversion
wxversion.select('2.8')
import wx

from matplotlib import use
use('WXAgg')

#wheee, monkey patches:
points_to_draw = []
import wxRayTrace.gui.glplotframe
def DrawPoint(self, point):
    rc, gc, bc=(1.0, 1.0, 1.0)
    size = 1.0
    glBegin(GL_LINES)
    glColor4f(rc, gc, bc, 1.)
    glVertex3f( point[0]+size, point[1], point[2])
    glVertex3f( point[0]-size, point[1], point[2])
    glVertex3f( point[0], point[1]+size, point[2])
    glVertex3f( point[0], point[1]-size, point[2])
    glVertex3f( point[0], point[1], point[2]+size)
    glVertex3f( point[0], point[1], point[2]-size)
    glEnd()

def new_draw(self):
    #Draw Rays
    #print "RRRRRRRRRRRRRRRRRRRRRRxR", self.os
    if self.os != None:
        for i in self.os.prop_ray:
                self.DrawRay(i)
        #Draw Components
        for comp in self.os.complist:
                self.DrawComp(comp)
        for point in points_to_draw:
                DrawPoint(self, point)
wxRayTrace.gui.glplotframe.glCanvas.DrawGLL = new_draw

from pyoptools.all import *
from pyoptools.misc import cmisc
from pyoptools.misc import pmisc
import scipy.integrate

import rotation_matrix
from meshsurface import MeshSurface


def Point2D(*args):
    return numpy.array(args)
Point3D = Point2D
AngleVector = namedtuple('AngleVector', ['theta', 'phi'])

class PointComponent(Component):
    """
    A bunch of squares, just for visualizing points
    """

    def _get_hitlist(self):
        return tuple()

    hit_list=property(_get_hitlist)

    def __init__(self, location, transparent=True,*args,**kwargs):
        Component.__init__(self, *args, **kwargs)
        size = numpy.array((1, 1))
        self.size=size
        self.location = location
        self.surflist["S1"] = (Plane(shape=Rectangular(size=size)), (0, 0, 0), (0, 0, 0))
        self.surflist["S2"] = (Plane(shape=Rectangular(size=size)), (0, 0, 0), (math.pi/2.0, 0, 0))
        self.surflist["S3"] = (Plane(shape=Rectangular(size=size)), (0, 0, 0), (0, math.pi/2.0, 0))
        self.material = 1.0

    def component(self):
        return (self, self.location, (0,0,0))

class ScreenComponent(Component):
    """
    A square to represent the screen surface
    """

    def _get_hitlist(self):
        return tuple(self.__d_surf.hit_list)

    hit_list=property(_get_hitlist)

    def __init__(self, size, transparent=True,*args,**kwargs):
        Component.__init__(self, *args, **kwargs)
        self.__d_surf= Plane(shape=Rectangular(size=size))
        self.size=size
        self.surflist["S1"]=(self.__d_surf,(0,0,0),(0,0,0))
        self.material=1.

def create_transform_matrix_from_rotations(rotations):
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

class Screen(object):
    """
    Creates some geometry for the screen, in the correct location.

    :attr position: the location of the center of the screen
    :type position: Point3D
    :attr rotations: (rx,ry,yz). Pretty dumb way of thinking about things but whatever.
    :type rotations: Point3D
    :attr size: the width and height of the screen
    :type size: Point2D
    :attr pixel_distribution:
    :type pixel_distribution: function(AngleVector) -> Point2D (distributed in the range [-1,-1] to [1,1])
    """

    def __init__(self, location, rotations, size, pixel_distribution):
        self.position = location
        self.rotations = rotations
        self.size = size
        self.pixel_distribution = pixel_distribution

        rot_mat = create_transform_matrix_from_rotations(self.rotations)

        self.side_vector = rot_mat.dot(Point3D(1.0, 0.0, 0.0))
        self.up_vector = rot_mat.dot(Point3D(0.0, 1.0, 0.0))
        self.direction = rot_mat.dot(Point3D(0.0, 0.0, -1.0))

    def vision_ray_to_pixel(self, vision_ray):
        """
        :param vision_ray: a vector from the eye
        :type  vision_ray: AngleVector
        :returns: location on the screen that corresponds to the given vision ray
        :rtype: Point
        """
        point = self.pixel_distribution(vision_ray)
        pixel = (point[0] * self.size[0]/2.0 * self.side_vector) + \
               (point[1] * self.size[1]/2.0 * self.up_vector) + \
               self.position
        return pixel

    def create_component(self):
        return (ScreenComponent(self.size), self.position, self.rotations)

def create_shell(distance, principal_eye_vector, radius, arcs):
    """
    Create a triangle mesh approximation of the screen
    """
    thickness = 5.0
    shape = Circular(radius=radius)
    front_surface = MeshSurface(arcs, shape=shape, reflectivity=1.0)
    component = Component(surflist=[(front_surface, (0, 0, 0), (0, 0, 0))], material=schott["BK7"])
    MirrorShell = namedtuple('MirrorShell', ['component', 'position', 'direction'])
    return MirrorShell(component, (0, 0, 0), (0, 0, 0))

def create_detector():
    """
    For now, just make a CCD oriented along the principal ray, centered at the center of the eye
    """
    #25 mm is approximately the size of the human eye
    ccd = CCD(size=(25, 25), transparent=False)
    Detector = namedtuple('Detector', ['ccd', 'position', 'direction'])
    return Detector(ccd, (0, 0, 0), (0, 0, 0))

def create_rays(screen, fov):
    """
    For now, just make a bunch of rays coming off of the screen.
    """
    return [Ray(pos=screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=screen.direction) \
            for (theta, phi) in \
            #[(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]
        [(0, 0)]]

#TODO: probably a good idea to cache these results...
def _get_theta_normal(principal_ray, theta):
    """
    Simply the normal to the plane defined by theta and the principal ray
    """
    base_principal_ray = Point3D(0.0, 0.0, -1.0)
    ray_rotation = numpy.zeros((3, 3))
    rotation_matrix.R_2vect(ray_rotation, base_principal_ray, principal_ray)
    base_theta_ray = Point3D(math.cos(theta), math.sin(theta), 0)
    return ray_rotation.dot(base_theta_ray)

def _normalize(a):
    return a / numpy.linalg.norm(a)

def _normalized_vector_angle(v1_u, v2_u):
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


def create_arc(screen, principal_ray, distance, theta):
    """
    Given a screen, principal ray, distance, and theta, calculate the shape of the screen such that
    every vision ray gives a correct reflection to the corresponding pixel.
    """

    #this function defines the derivative of the surface at any given point.
    #simply the intersection of the theta plane and the required surface plane at that point
    #if there is no way to bounce to the front of the screen, the derivative is just 0
    theta_normal = _get_theta_normal(principal_ray, theta)
    def f(point, t):
        #TODO: return [0,0,0] if the point is not in front of the screen (since it would not be visible at all, we should stop tracing this surface)
        eye_to_point_vec = _normalize(point)
        phi = _normalized_vector_angle(principal_ray, eye_to_point_vec)
        pixel_point = screen.vision_ray_to_pixel(AngleVector(theta, phi))
        point_to_eye_vec = eye_to_point_vec * -1
        point_to_screen_vec = _normalize(pixel_point - point)
        surface_normal = _normalize((point_to_screen_vec + point_to_eye_vec) / 2.0)
        #TODO: might want to reverse the order of these just to be more intuitive. I feel like positive t should move from the center outward
        derivative = numpy.cross(surface_normal, theta_normal)
        return derivative

    #initial point is distance away, along the principal ray
    point0 = principal_ray * distance

    #the set of t values for which we would like to have output.
    #t is a measure of distance along the surface of the screen.
    #TODO: could do a better job of estimating how much t is required
    #more t means that we're wasting time, less t means we might not quite finish defining the whole surface
    #could probably be much more intelligent about this
    t_step = 0.1
    max_t = 100
    t_values = numpy.arange(0, max_t, t_step)

    #actually perform the numerical integration.
    result = scipy.integrate.odeint(f, point0, t_values)
    #convert the result into a more useful form
    #result = result[:, 0]

    return result

def main():
    #create the main app
    app = wx.PySimpleApp()

    #system assumptions
    fov = math.pi / 4.0
    screen_angle = math.pi / 8.0
    principal_eye_vector = Point3D(0.0, 0.0, -1.0)

    #create the components
    screen_location = Point3D(0, 40.0, -20.0)
    screen_rotation = Point3D(-screen_angle, 0, 0)
    screen_size = Point2D(25.0, 25.0)
    def pixel_distribution(vec):
        r = vec.phi / fov
        return Point2D(r*math.cos(vec.theta), r*math.sin(vec.theta))
    screen = Screen(screen_location, screen_rotation, screen_size, pixel_distribution)

    shell_distance = 60.0
    shell_radius = 40.0#80.0

    #create number of different arcs along the surface (for debugging this function)
    arcs = []
    for arc_theta in numpy.arange(0, 2.0 * math.pi, math.pi / 200.0):
        arc = create_arc(screen, principal_eye_vector, shell_distance, arc_theta)
        #visualize the arc
        #arc = arc[0::10]
        arcs.append(arc)
        #for point in arc:
        #    points_to_draw.append(point)

    shell = create_shell(shell_distance, principal_eye_vector, shell_radius, arcs)
    detector = create_detector()
    raylist = create_rays(screen, fov)


    #assemble them into the system
    system = System(complist=[screen.create_component(), shell, detector], n=1)
    system.ray_add(raylist)

    #run the simulation
    system.propagate()
    glPlotFrame(system)
    spot_diagram(detector.ccd)
    app.MainLoop()

if __name__ == '__main__':
    main()

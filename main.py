#!/user/bin/python

"""
Installation:
sudo apt-get install python-wxgtk2.8 python-vtk

A tool for exploring freeform optics in the context of a laser raster based head mounted display.

Specifically, generates the reflective shell based on a variety of system parameters, then traces
rays through the system.

Can generate a resulting spot diagram.

Also visualizes the system with a crappy OpenGL wx wrapper.

At a high level, there are only 3 components in the system--the screen (kind of mounted on your
forehead), the reflective shell (in front of you, reflecting those rays to your eye), and your eye.

Assumptions:
    all units are mm and radians
    center of eyeball at (0, 0, 0) facing -z
    top of head towards +y
    right is +x
definitions:
    lightRay: a ray coming from a pixel on the screen towards the eye
    visionRay: a ray coming from the eye at a particular (theta, phi)
    principalRay: the visionRay with theta=0 and phi=0
    for all rays:
        theta: here, theta is the rotational angle in the 2D plane that you are viewing with your eye as if it were a graph.
                Thus, 0 is to the right, and increasing theta moves you counter clockwise
        phi: how much off-axis the ray is going. 0 is straight ahead (along the principal ray), increasing phi is increasingly off-axis
define system parameters:
    fieldOfViewFunction: f(theta) = phi. Defines how wide the field of view is at a given angle
    screen
        center: (x, y, z) of screen center
        angle: positive angle is tilted towards eye
        width: dimension in x (when angle is 0)
        height: dimension in y (when angle is 0)
        pixelDistribution: defines a bidirectional mapping between visionRays and (x,y,z) coordinates
    shell
        distance: distance away from eye center (along the -z axis). This completely defines the screen.
        coefficients: the coefficients of the taylor poly with which we are approximating the surface.
            May replace this with a more numerical approach. Taylor poly is a function in x/y as defined above.
core singular modeling assumption (for now):
    pixelDistribution makes sense, eg, that there is such a 1:1 mapping between locations on the screen and (theta, phi) pairs (eg, vision rays)
        this basically transforms the problem into one of figuring out the pixelDistribution (and eventually, screen shape) instead of the surface
    also--the surface should be smooth and continuous (micropixels might allow something slightly better, but no)

Given that, how it works now is relatively simple:
    (pick values for all parameters above)
    pick a distance away from the eye (along the principal ray)
    for various values of theta in the range [0, 2pi):
        let the thetaPlane be the YZ-plane rotated about the principalRay by theta
        define a vector field (ODE) based on the fact that at any given point, we know theta (obviously) and can thus calculate phi, which means we can calculate the position of the pixel, which means we can calculate the angle required for the screen surface to bounce the ray from the eye to the pixel
        do numerical integration to walk through that field along the thetaPlane (starting at the point along the central ray that is "distance" away from the eye)
        this walk is an "arc"--a series of points that lie on the surface of the reflective shell
    after all arcs have been defined, simply connect the together in a triangle mesh in the obvious way

references:
    https://pyoptools.googlecode.com/hg-history/stable/doc/_build/html/pyoptools_raytrace_surface.html#pyoptools.raytrace.surface.TaylorPoly
    http://www.creol.ucf.edu/research/publications/2012.pdf
    https://gitorious.org/vtkwikiexamples/wikiexamples/raw/a457b8a51c3084896f98a0acd4faf6bb2a88031e:Python/PolyData/SmoothMeshgrid.py
    https://bitbucket.org/somada141/pycaster/src/50512b958dc5bc91de7ae3f532a650950c68ab10/pycaster/pycaster.py?at=master
"""

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

PLANE_ANGLE_STEP = math.pi / 20.0


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
        pixel = (point[0] * self.size[0]/2.0 * self.side_vector * -1.0) + \
               (point[1] * self.size[1]/2.0 * self.up_vector * -1.0) + \
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
    #return [Ray(pos=screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=screen.direction) \
    #        for (theta, phi) in \
    #        #[(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]
    #    [(0, 0)]]

    #fov = fov*0.99
    #return [Ray(pos=ray, dir=ray) for ray in [(0.0, 0.0, -1.0),
    #    (math.sin(fov), 0.0, -math.cos(fov)),
    #    (-math.sin(fov), 0.0, -math.cos(fov)),
    #    (0.0, math.sin(fov), -math.cos(fov)),
    #    (0.0, -math.sin(fov), -math.cos(fov))]]

    rays = []
    #tup = (math.sin(fov), 0.0, -math.cos(fov))
    #rays.append(Ray(pos=tup, dir=tup))
    for i in range(0, 100):
        angle_step = (fov * 2.0) / 100.0
        angle = -fov + angle_step*float(i)
        #horizontal fan of rays
        #tup = (math.sin(angle), 0.0, -math.cos(angle))
        #vertical fan of rays
        tup = (0.0, math.sin(angle), -math.cos(angle))
        rays.append(Ray(pos=(0,0,0), dir=tup))
    return rays

#TODO: probably a good idea to cache these results...
def _get_theta_normal(principal_ray, theta):
    """
    Simply the normal to the plane defined by theta and the principal ray
    """
    base_principal_ray = Point3D(0.0, 0.0, -1.0)
    ray_rotation = numpy.zeros((3, 3))
    rotation_matrix.R_2vect(ray_rotation, base_principal_ray, principal_ray)
    base_theta_ray = Point3D(-math.sin(theta), math.cos(theta), 0)
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
        derivative = _normalize(numpy.cross(surface_normal, theta_normal))
        return derivative

    #initial point is distance away, along the principal ray
    point0 = principal_ray * distance

    #the set of t values for which we would like to have output.
    #t is a measure of distance along the surface of the screen.
    #TODO: could do a better job of estimating how much t is required
    #more t means that we're wasting time, less t means we might not quite finish defining the whole surface
    #could probably be much more intelligent about this
    t_step = 0.2
    max_t = 100
    t_values = numpy.array(list(numpy.arange(0, 10.0, 0.5)) + list(numpy.arange(10.1, max_t, t_step)))

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
    shell_radius = 60.0#80.0

    #create number of different arcs along the surface (for debugging this function)
    arcs = []
    for arc_theta in numpy.arange(0, 2.0 * math.pi, PLANE_ANGLE_STEP):
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

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
core singular modeling assumption (for now):
    pixelDistribution makes sense, eg, that there is such a 1:1 mapping between locations on the screen and (theta, phi) pairs (eg, vision rays)
        this basically transforms the problem into one of figuring out the pixelDistribution (and eventually, screen shape) instead of the surface
    also--the surface should be smooth and continuous (micropixels might allow something slightly better, but no)

Given that, how it works now is relatively simple:
    (pick values for all parameters above)
    pick a distance away from the eye (along the principal ray)
    the point that is that distance away from the eye along the principal ray uniquely defines the surface
    create an horizontal arc by walking through the vector field away from that point in both directions
    this is the "spine"
    then create a bunch of ribs by doing the same process to create vertical arcs (one at each point in the spine)
    those vertical ribs can be linked together to form a triangle mesh which we can then draw, bounce rays off, etc
    
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

#wheee, monkey patches. simply made it so that any points in points_to_draw will be rendered as a little cross (3 intersecting lines)
points_to_draw = []
import wxRayTrace.gui.glplotframe
def DrawPoint(self, point):
    """
    Draws a point as 3 intersecting, small lines. Why? Because I don't want to mess around with GL_POINTs and the associated parameters to color and size them
    
    Feel free to change
    """
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
    """
    replacement of the original function so that we can also draw debug points (in addition to rays and components)
    """
    if self.os != None:
        for i in self.os.prop_ray:
                self.DrawRay(i)
        for comp in self.os.complist:
                self.DrawComp(comp)
        #Monkey patched to add this--just want to draw all of the points for easy debugging
        for point in points_to_draw:
                DrawPoint(self, point)
wxRayTrace.gui.glplotframe.glCanvas.DrawGLL = new_draw

from pyoptools.all import *
from pyoptools.misc import cmisc
from pyoptools.misc import pmisc
import scipy.integrate

import rotation_matrix
from meshsurface import MeshSurface

#these are very simple classes. just an alias for a numpy array that gives a little more intentionality to the code
def Point2D(*args):
    return numpy.array(args)
def Point3D(*args):
    return numpy.array(args)
#Used for vision and light rays. See module docstring
AngleVector = namedtuple('AngleVector', ['theta', 'phi'])

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
    """
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

class Screen(object):
    """
    Creates some geometry for the screen, in the correct location.

    :attr position: the location of the center of the screen
    :type position: Point3D
    :attr rotations: (rx,ry,yz). Pretty dumb way of thinking about things but whatever.
    see 'create_transform_matrix_from_rotations' for a description of the format
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
        Convert from a VisionRay into a Point3D corresponding to where the pixel would be in real space on the screen
        
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
        """
        Returns a Component for pyOpTools
        """
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

def create_rays_from_screen(screen, fov):
    """
    For now, just make a bunch of light rays coming off of the screen.
    """
    return [Ray(pos=screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=screen.direction) \
            for (theta, phi) in \
            [(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]

def create_rays(screen, fov):
    """
    Create a bunch of vision rays coming from your eye
    """
    rays = []
    for i in range(0, 100):
        angle_step = (fov * 2.0) / 100.0
        angle = -fov + angle_step*float(i)
        #horizontal fan of rays
        tup = (math.sin(angle), 0.0, -math.cos(angle))
        #vertical fan of rays
        #tup = (0.0, math.sin(angle), -math.cos(angle))
        rays.append(Ray(pos=(0,0,0), dir=tup))
    return rays

def _get_arc_plane_normal(principal_ray, is_horizontal):
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

def _get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, point):
    #project point onto the p=(0,0,0),n=principal_ray plane
    dist = principal_ray.dot(point)
    projected_point = point - principal_ray*dist
    #normalize. if 0 length, return 0
    length = numpy.linalg.norm(projected_point)
    if length == 0.0:
        return 0.0
    normalized_point = projected_point / length
    #measure angle between normalized projection and v_arc_normal
    theta = _normalized_vector_angle(normalized_point, v_arc_normal)
    #if angle between normalized projection and h_arc_normal is > pi / 2.0, subtract angle from 2.0 * pi
    if _normalized_vector_angle(normalized_point, h_arc_normal) > math.pi / 2.0:
        theta = math.pi * 2.0 - theta
    return theta

#TODO: this code with any non-(0,0,-1) principal ray is all untested. 
def create_new_arc(screen, principal_ray, point0, is_horizontal=None):
    """
    Given a screen and point, calculate the shape of the screen such that
    every vision ray gives a correct reflection to the corresponding pixel.
    
    Can either trace 'horizontally' or 'vertically' (relative to the principal ray)
    """
    
    assert is_horizontal != None
    h_arc_normal = _get_arc_plane_normal(principal_ray, True)
    v_arc_normal = _get_arc_plane_normal(principal_ray, False)

    #this function defines the derivative of the surface at any given point.
    #simply the intersection of the arc plane and the required surface plane at that point
    #if there is no way to bounce to the front of the screen, the derivative is just 0
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    def f(point, t):
        #TODO: return [0,0,0] if the point is not in front of the screen (since it would not be visible at all, we should stop tracing this surface)
        eye_to_point_vec = _normalize(point)
        phi = _normalized_vector_angle(principal_ray, eye_to_point_vec)
        theta = _get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, point)
        pixel_point = screen.vision_ray_to_pixel(AngleVector(theta, phi))
        point_to_eye_vec = eye_to_point_vec * -1
        point_to_screen_vec = _normalize(pixel_point - point)
        surface_normal = _normalize((point_to_screen_vec + point_to_eye_vec) / 2.0)
        derivative = _normalize(numpy.cross(surface_normal, arc_plane_normal))
        return derivative

    #the set of t values for which we would like to have output.
    #t is a measure of distance along the surface of the screen.
    #TODO: could do a better job of estimating how much t is required
    #more t means that we're wasting time, less t means we might not quite finish defining the whole surface
    #could probably be much more intelligent about this
    t_step = 1.0
    max_t = 80.0
    t_values = numpy.arange(0.0, max_t, t_step)

    #actually perform the numerical integration.
    half_arc = scipy.integrate.odeint(f, point0, t_values)
    
    #do the other half as well:
    def g(point, t):
        return -1.0 * f(point, t)
    
    #actually perform the numerical integration.
    other_half_arc = list(scipy.integrate.odeint(g, point0, t_values))
    other_half_arc.pop(0)
    other_half_arc.reverse()
    result = other_half_arc + list(half_arc)
    
    return numpy.array(result)

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
    shell_radius = 60.0
    
    #create the main arc
    starting_point = principal_eye_vector * shell_distance
    spine = create_new_arc(screen, principal_eye_vector, starting_point, is_horizontal=True)
    #create each of the arcs reaching off of the spine
    arcs = []
    for point in spine:
        arc = create_new_arc(screen, principal_eye_vector, point, is_horizontal=False)
        arcs.append(arc)
        
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

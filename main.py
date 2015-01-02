#!/user/bin/python

"""
DEPRECATED: transitioning all of this functionality to interface.py instead.
Migration plan: all common code will be refactored out into optics.py
Feel free to fiddle with anything in this script, but not things in optics.py
Goal: eventually would like to stop using pyoptools entirely
    For now, the short term goal is to remove all dependency on their wxRayTrace module at the very least
    This will give us full control over the interface
    Should be pretty easy to render all of this stuff in our own interface instead of theirs
We can keep some of the other bits about rays, components, etc, because we might end up putting the entire thing in C
If we don't put it in C, we might end up using Cython, and then we can probably keep what they have

Installation:
sudo apt-get install python-wxgtk2.8 python-vtk cython
sudo apt-get install subversion
sudo apt-get install ipython build-essential python-numpy python-scipy python-matplotlib python-wxversion python-dev python-opengl
svn checkout http://pyoptools.googlecode.com/svn/trunk/ pyoptools
cd pyoptools
sudo python setup.py install

(instructions mostly from here http://pyoptools.googlecode.com/hg-history/c72e8881fda92f41754d668819c93b688df8c62c/doc/_build/html/install.html )

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
    lightRay: a ray coming from a pixel on the screen towards the eye. defined as an (x,y,z) vector of length 1
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
global_rays = []
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
import mesh
from meshsurface import MeshSurface

from optics import Point3D, Point2D, _normalize, _normalized_vector_angle, _get_arc_plane_normal, angle_vector_to_vector, AngleVector, create_transform_matrix_from_rotations

FOCAL_LENGTH = 24.0
PUPIL = 4

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

MirrorShell = namedtuple('MirrorShell', ['component', 'position', 'rotations'])

class ExternalMeshSurface(MeshSurface):
    def __init__(self, filename, *args, **kwargs):
        Surface.__init__(self, *args, **kwargs)
        self._mesh = mesh.Mesh(mesh.load_stl(filename))
        self._inf_vector = numpy.array((numpy.inf,numpy.inf,numpy.inf))
        
def create_shell(distance, principal_eye_vector, radius, arcs):
    """
    Create a triangle mesh approximation of the screen
    """
    thickness = 5.0
    shape = Circular(radius=radius)
    front_surface = MeshSurface(arcs, shape=shape, reflectivity=1.0)
    component = Component(surflist=[(front_surface, (0, 0, 0), (0, 0, 0))], material=schott["BK7"])
    return MirrorShell(component, (0, 0, 0), (0, 0, 0))

def load_shell(filename, radius):
    """
    Load a triangle mesh approximation of the screen from a file
    """
    shape = Circular(radius=radius)
    front_surface = ExternalMeshSurface(filename, shape=shape, reflectivity=1.0)
    component = Component(surflist=[(front_surface, (0, 0, 0), (0, 0, 0))], material=schott["BK7"])
    return MirrorShell(component, (0, 0, 0), (0, 0, 0))

def create_detector(size, position, rotations):
    """
    For now, just make a CCD oriented along the principal ray, centered at the center of the eye
    size is a tuple of length by width

    Returns surflist for Component class, in the form of (surface, (posX,posY,posZ), (rotX,rotY,rotZ))
    """
    #25 mm is approximately the size of the human eye
#    ccd = CCD(size=(25, 25), transparent=False)
    ccd = CCD(size=size, transparent=False)
    Detector = namedtuple('Detector', ['ccd', 'position', 'rotations'])
#    return Detector(ccd, (0, 0, FOCAL_LENGTH/2.0), (0, 0, 0))
    return Detector(ccd, position, rotations)

def create_cornea():
    surface_radius = 14.4
    lens_radius = 2 * PUPIL
    offset = surface_radius - math.sqrt(surface_radius*surface_radius-lens_radius*lens_radius)
    surface1 = Spherical(curvature=1.0/surface_radius, shape=Circular(radius=lens_radius))
    surface2 = Spherical(curvature=1.0/surface_radius, shape=Circular(radius=lens_radius))
    lens = Component(surflist=[(surface1, (0, 0, -offset), (0, 0, 0)), (surface2, (0, 0, offset), (0, math.pi, 0))], material=1.3)
    Cornea = namedtuple('Cornea', ['lens', 'position', 'rotations'])
    return Cornea(lens, (0, 0, -FOCAL_LENGTH/2.0), (0, 0, 0))

def create_iris():
    surface = Aperture(shape=Rectangular(size=(150,150)), ap_shape=Circular(radius=PUPIL))
    aperture = Component(surflist=[(surface, (0, 0, 0), (0, 0, 0))])
    Iris = namedtuple('Iris', ['aperture', 'position', 'rotations'])
    return Iris(aperture, (0, 0, -FOCAL_LENGTH/2.0), (0, 0, 0))

def create_eye(look_direction):
    """
    Creates an eye that looks in the direction of look_direction.

    attr look_direction: 
    type 
    
    Returns tuple of 
    """
    iris = create_iris()
    cornea = create_cornea()

    return (iris, cornea)
    

def create_rays_from_screen(screen, fov):
    """
    For now, just make a bunch of light rays coming off of the screen.
    """
    (theta, phi) = (0,0)
    rotations = []
    rays = []

    for xangle in numpy.arange(-fov/10, fov/10, fov/80):
        for yangle in numpy.arange(-fov/10, fov/10, fov/80):
            rotations.append((xangle, yangle, 0))

    for rot in rotations:
        rot_mat = create_transform_matrix_from_rotations(rot)
        rays.append(Ray(screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=rot_mat.dot(screen.direction)))

    return rays

    
#    return [Ray(pos=screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=screen.direction) \
#            for (theta, phi) in \
#            [(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]

def create_rays_parallel_to_eye(screen, fov):
    rays = []
    x = 0.0
    #for x in numpy.arange(-PUPIL, PUPIL, 0.2):
    for y in numpy.arange(-PUPIL, PUPIL, 0.2):
        rays.append( Ray(pos=(x, y, -10), dir=(0, 0, -1)) )
    return rays

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
    
    def get_center(x, x_min, x_max, x_step):
        centers = numpy.arange(x_min, x_max, x_step)
        # if surface is only divided into one piece, return that center
        if len(centers) == 1:
            return centers[0], 0
        for i in range(0, len(centers)-1):
            prev = centers[i]
            next = centers[i+1]
            # if x falls within a boundary, return the previous center
            if x < (next + prev) / 2.0:
                return prev, i 
        # if x is not less than any boundary, use last center
        return centers[-1], len(centers)-1
            
    def find_center_given_phi_theta(phi, theta):
        fov = math.pi/4
        bucketed_phi, bucket_num = get_center(phi, 0, fov, fov/10.0)
        num_theta_buckets = ((bucket_num) * 2) + 1
        bucketed_theta, theta_index = get_center(theta, 0, 2.0*math.pi, 2.0*math.pi/float(num_theta_buckets))
        return bucketed_phi, bucketed_theta
    

    #this function defines the derivative of the surface at any given point.
    #simply the intersection of the arc plane and the required surface plane at that point
    #if there is no way to bounce to the front of the screen, the derivative is just 0
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    def f(point, t):
        #TODO: return [0,0,0] if the point is not in front of the screen (since it would not be visible at all, we should stop tracing this surface)

        # This section creates surface such that a ray from center of the eye hits the correct pixel on the screen
        eye_to_point_vec = _normalize(point)
        phi = _normalized_vector_angle(principal_ray, eye_to_point_vec)
        theta = _get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, point)

        # This section creates surface such that a cone of rays coming from a pixel on the screen is reflected in parallel rays toward the eye
        # Assumes that screen is tilted at correct angle such that ray coming from eye center hits screen at perpendicular angle
#        eye_to_point_vec = _normalize(point)
#        phi = _normalized_vector_angle(principal_ray, eye_to_point_vec)
#        theta = _get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, point)
#        phi, theta = find_center_given_phi_theta(phi, theta)
#        eye_to_point_vec = angle_vector_to_vector(AngleVector(theta, phi), principal_ray)
        
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

def broken_create_arc(screen, principal_ray, shell_point, is_horizontal=True):
#def broken_create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=None):
    assert is_horizontal != None, "Must pass this parameter"
    angle_vec = AngleVector(0.0, 0.0)
    screen_point = screen.position
    
    ##HACK: just seeing what happens if I do this:
    #shell_to_screen_vec = screen_point - shell_point
    #screen_point = 20.0 * shell_to_screen_vec + shell_point
    
    #define a vector field for the surface normals of the shell.
    #They are completely constrained given the location of the pixel and the fact
    #that the reflecting ray must be at a particular angle        
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    desired_light_direction_off_screen_towards_eye = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    def f(point, t):
        point_to_screen_vec = _normalize(screen_point - point)
        surface_normal = _normalize(point_to_screen_vec + desired_light_direction_off_screen_towards_eye)
        derivative = _normalize(numpy.cross(surface_normal, arc_plane_normal))
        return derivative
    
    #TODO: this should really be based on light_radius...
    
    #estimate how long the piece of the shell will be (the one that is large enough to reflect all rays)
    #overestimates will waste time, underestimates cause it to crash :-P
    #note that we're doing this one half at a time
    def estimate_t_values():
        #TODO: make this faster if necessary by doing the following:
            #define the simple line that reflects the primary ray
            #intersect that with the max and min rays from the eye
            #check the distance between those intersections and double it or something
        t_step = 0.5
        #if LOW_QUALITY_MODE:
        #    t_step = 0.5
        max_t = 5.0
        return numpy.arange(0.0, max_t, t_step)
    t_values = estimate_t_values()

    #use the vector field to define the exact shape of the surface (first half)
    half_arc = scipy.integrate.odeint(f, shell_point, t_values)
    
    #do the other half as well
    def g(point, t):
        return -1.0 * f(point, t)
    
    #combine them
    other_half_arc = list(scipy.integrate.odeint(g, shell_point, t_values))
    other_half_arc.pop(0)
    other_half_arc.reverse()
    return other_half_arc + list(half_arc)

def main_3d():
    #create the main app
    app = wx.PySimpleApp()

    #system assumptions
    fov = math.pi / 4.0
    screen_angle = math.pi / 4.0
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
    
    ##create the main arc
    #starting_point = principal_eye_vector * shell_distance
    #spine = create_new_arc(screen, principal_eye_vector, starting_point, is_horizontal=True)
    ##create each of the arcs reaching off of the spine
    #arcs = []
    #for point in spine:
    #    arc = create_new_arc(screen, principal_eye_vector, point, is_horizontal=False)
    #    arcs.append(arc)
    #shell = create_shell(shell_distance, principal_eye_vector, shell_radius, arcs)
    
    #create the main arc
    starting_point = principal_eye_vector * shell_distance
    spine = broken_create_arc(screen, principal_eye_vector, starting_point, is_horizontal=True)
    #create each of the arcs reaching off of the spine
    arcs = []
    for point in spine:
        arc = broken_create_arc(screen, principal_eye_vector, point, is_horizontal=False)
        arcs.append(arc)
    shell = create_shell(shell_distance, principal_eye_vector, shell_radius, arcs)
    
    #shell = load_shell("all_scales.stl", shell_radius)
    
    detector = create_detector((500,500), screen_location*3, screen_rotation)
    raylist = create_rays_parallel_to_eye(screen, fov)

    # add ray shooting out of center of eye for debugging
    raylist.append(Ray(pos=(0, 0, 0), dir=(0, 0, -1)))


    #raylist = []
    #iris = create_iris()
    #cornea = create_cornea()

    #assemble them into the system
    #system = System(complist=[screen.create_component(), shell, detector, cornea, iris], n=1)
    system = System(complist=[screen.create_component(), shell, detector], n=1)
    system.ray_add(raylist)

    #run the simulation
    system.propagate()
    glPlotFrame(system)
    spot_diagram(detector.ccd)
    app.MainLoop()

#TODO: we should probably delete this 2D duplicated code?
############################################################
# Drawing rays and surface in 2D to better understand optics
############################################################

def create_rays_from_screen_center_2d(screen, fov):
    """
    For now, just make a bunch of light rays coming off of the screen.
    """
    (theta, phi) = (0,0)
    rotations = []
    rays = []
    for angle in numpy.arange(fov/8, fov/4, fov/32):
        rotations.append((angle, 0, 0))

    for rot in rotations:
        rot_mat = create_transform_matrix_from_rotations(rot)
        rays.append(Ray(screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=rot_mat.dot(screen.direction)))
    return rays

def create_rays_from_multiple_pixels_on_screen_2d(screen, fov):
    """
    Make light rays go in all directions from multiple pixels on screen.
    """
    # list of pixels based on theta, phi values
    pixel_list = [
        (0, 0),
        (0, fov/2),
        (0, fov)
        ]
    rotations = []
    rays = []

    for angle in numpy.arange(-fov/2, fov/2, fov/40):
        rotations.append((angle, 0, 0))

    for theta, phi in pixel_list:
        for rot in rotations:
            rot_mat = create_transform_matrix_from_rotations(rot)
            rays.append(Ray(screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=rot_mat.dot(screen.direction)))
    return rays


def create_parallel_rays_from_eye_to_screen_2d(screen, fov):
    """
    Creates a few sets of parallel rays from the center of the eye to the screen.
    Need to turn off eye lens to use this properly.

    This helps determine where point source on screen should be relative to shell surface.
    """
    origin = Point3D(0, 0, 0)
    rotations = []
    rays = []
    principal_eye_rays = [
        Ray(pos = origin, dir = (0, 0, -1)),
        Ray(pos = origin, dir = (0, 1, -1)),
        Ray(pos = origin, dir = (0, 0.5, -1)),
        Ray(pos = origin, dir = (0, 0.2, -1)),        
        Ray(pos = origin, dir = (0, -0.2, -1)),
        ]
    
    for ray in principal_eye_rays:
        for x in numpy.arange(1, 3, 0.5):
            rays.append(Ray(pos = origin+Point3D(0,x,0), dir = ray.dir))
            rays.append(Ray(pos = origin+Point3D(0,-x,0), dir = ray.dir))

    return rays + principal_eye_rays
        

#TODO: this code with any non-(0,0,-1) principal ray is all untested. 
def create_new_arc_2d(screen, principal_ray, point0, is_horizontal=None):
    """
    Given a screen and point, calculate the shape of the screen such that
    every vision ray gives a correct reflection to the corresponding pixel.
    
    Can either trace 'horizontally' or 'vertically' (relative to the principal ray)
    """
    
    assert is_horizontal != None
    h_arc_normal = _get_arc_plane_normal(principal_ray, True)
    v_arc_normal = _get_arc_plane_normal(principal_ray, False)

    def get_center(x, x_min, x_max, x_step):
        centers = numpy.arange(x_min, x_max, x_step)
        # if surface is only divided into one piece, return that center
        if len(centers) == 1:
            return centers[0], 0
        for i in range(0, len(centers)-1):
            prev = centers[i]
            next = centers[i+1]
            # if x falls within a boundary, return the previous center
            if x < (next + prev) / 2.0:
                return prev, i 
        # if x is not less than any boundary, use last center
        return centers[-1], len(centers)-1
            
    def find_center_given_phi_theta(phi, theta):
        fov = math.pi/4
        bucketed_phi, bucket_num = get_center(phi, 0, fov, fov/30.0)
        #num_theta_buckets = ((bucket_num) * 4) + 1
        #bucketed_theta, theta_index = get_center(theta, 0, 2.0*math.pi, 2.0*math.pi/float(num_theta_buckets))
        bucketed_theta = math.pi / 2.0
        if theta > math.pi:
            bucketed_theta = 3.0 * math.pi / 2.0
        return bucketed_phi, bucketed_theta
    
    #this function defines the derivative of the surface at any given point.
    #simply the intersection of the arc plane and the required surface plane at that point
    #if there is no way to bounce to the front of the screen, the derivative is just 0
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    def f(point, t):
        # This section creates surface such that a ray from center of the eye hits the correct pixel on the screen
#        eye_to_point_vec = Point3D(0, 0, -1)
        eye_to_point_vec = _normalize(point)
        phi = _normalized_vector_angle(principal_ray, eye_to_point_vec)
        theta = _get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, point)
        bucketed_phi, bucketed_theta = find_center_given_phi_theta(phi, theta)
        eye_to_point_vec = angle_vector_to_vector(AngleVector(bucketed_theta, bucketed_phi), principal_ray)
        
        pixel_point = screen.vision_ray_to_pixel(AngleVector(bucketed_theta, bucketed_phi))
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
    t_step = 0.01
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


def main_2d():
    #create the main app
    app = wx.PySimpleApp()

    #system assumptions
    fov = math.pi / 4.0
    screen_angle = math.pi / 4.0
    principal_eye_vector = Point3D(0.0, 0.0, -1.0)

    #create the components
    screen_location = Point3D(0, 40.0, -20.0)#Point3D(0.0, 0.0, 0.0)
    screen_rotation = Point3D(-screen_angle, 0, 0)
    screen_size = Point2D(25.0, 25.0)
    def pixel_distribution(vec):
        r = vec.phi / fov
        # confused by your assumption here, pretty sure +theta goes toward +x from
        # +y axis...should be (r*math.sin(vec.theta), r*math.cos(vec.theta))
        # i flipped it here, fixed for me, change back if i'm wrong
        return Point2D(r*math.cos(vec.theta), r*math.sin(vec.theta))
    screen = Screen(screen_location, screen_rotation, screen_size, pixel_distribution)

    shell_distance = 80.0
    shell_radius = 100.0
    
    # create the starting point, with 3 points so we actually have a surface
    starting_point = principal_eye_vector * shell_distance
    spine = numpy.array([
        starting_point + Point3D(-1, 0, 0),
        starting_point,
        starting_point + Point3D(1, 0, 0),        
        ])
    #create each of the arcs reaching off of the spine
    arcs = []
    for point in spine:
        arc = create_new_arc_2d(screen, principal_eye_vector, point, is_horizontal=False)
        arcs.append(arc)
        
    shell = create_shell(shell_distance, principal_eye_vector, shell_radius, arcs)
    detector = create_detector((500,500), screen_location*3, screen_rotation)
#    detector = create_detector((25,25), (0,0,FOCAL_LENGTH/2.0), (0,0,0))
    raylist = create_parallel_rays_from_eye_to_screen_2d(screen, fov)

    iris = create_iris()
    cornea = create_cornea()

    #assemble them into the system
#    system = System(complist=[screen.create_component(), shell, detector, cornea, iris], n=1)
    system = System(complist=[screen.create_component(), shell, detector], n=1)
    system.ray_add(raylist)

    #run the simulation
    system.propagate()

    # print reflected rays
    for ray in system.prop_ray:
        for child in ray.childs:
            print child.dir
    
    glPlotFrame(system)
    spot_diagram(detector.ccd)
    app.MainLoop()    
    


if __name__ == '__main__':
    main_3d()

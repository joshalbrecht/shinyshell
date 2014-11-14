#!/user/bin/python

import math
from collections import namedtuple

import numpy

import wxversion
wxversion.select('2.8')
import wx

from matplotlib import use
use('WXAgg')

from pyoptools.all import *
from pyoptools.misc import cmisc
from pyoptools.misc import pmisc

import rotation_matrix


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

def create_shell(distance, principal_eye_vector, radius):
    """
    Create a very basic reflective screen that shoots rays in approximately the right direction
    """
    thickness = 5.0
    shape = Circular(radius=radius)
    cohef = numpy.array([[0, 0, 0.0039],[0, 0, 0],[0.0034, 0, 0]]).copy(order='C')
    front_surface = TaylorPoly(shape=shape, cohef=cohef, reflectivity=1.0)
    component = Component(surflist=[(front_surface, (0, 0, 0), (0, 0, 0))], material=schott["BK7"])
    MirrorShell = namedtuple('MirrorShell', ['component', 'position', 'direction'])
    return MirrorShell(component, (0, 0, -distance), (0, 0, 0))

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
            [(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]

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
    shell_radius = 80.0
    shell = create_shell(shell_distance, principal_eye_vector, shell_radius)
    detector = create_detector()
    raylist = create_rays(screen, fov)

    #assemble them into the system
    system = System(complist=[screen.create_component(), shell, detector,
                              PointComponent((10, 10, 0)).component(),
                              PointComponent((20, 20, 0)).component(),
                              PointComponent((30, 30, 0)).component()
                              ], n=1)
    system.ray_add(raylist)

    #run the simulation
    system.propagate()
    glPlotFrame(system)
    spot_diagram(detector.ccd)
    app.MainLoop()

if __name__ == '__main__':
    main()

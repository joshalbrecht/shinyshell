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

import rotation_matrix

def Point2D(*args):
    return numpy.array(args)
Point3D = Point2D
AngleVector = namedtuple('AngleVector', ['theta', 'phi'])

class Screen(object):
    """
    Creates some geometry for the screen, in the correct location.

    :attr position: the location of the center of the screen
    :type position: Point3D
    :attr direction: the direction the screen is facing
    :type direction: Point3D
    :attr size: the width and height of the screen
    :type size: Point2D
    :attr pixel_distribution:
    :type pixel_distribution: function(AngleVector) -> Point2D (distributed in the range [-1,-1] to [1,1])
    """

    def __init__(self, location, direction, size, pixel_distribution):
        self.position = location
        self.direction = direction
        self.size = size
        self.pixel_distribution = pixel_distribution

        original_direction = Point3D(0.0, 0.0, -1.0)
        rot_mat = numpy.array((
            numpy.array((0.0, 0.0, 0.0)),
            numpy.array((0.0, 0.0, 0.0)),
            numpy.array((0.0, 0.0, 0.0))
            ))
        rotation_matrix.R_2vect(rot_mat, original_direction, direction)
        self.side_vector = rot_mat.dot(Point3D(1.0, 0.0, 0.0))
        self.up_vector = rot_mat.dot(Point3D(0.0, 1.0, 0.0))

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

def create_shell(distance, principal_eye_vector, radius):
    """
    Create a very basic reflective screen that shoots rays in approximately the right direction
    """
    thickness = 5.0
    shape = Circular(radius=radius)
    cohef = numpy.array([[0,0.0001],[0.0001,0.0002]]).copy(order='C')
    front_surface = TaylorPoly(shape=shape, cohef=cohef, reflectivity=1.0)
    #rear_surface = TaylorPoly(shape=shape, cohef=cohef, reflectivity=1.0)
    #edge = Cylinder(radius=radius,length=thickness)
    #surflist=[(front_surface, (0, 0, 0),             (0, 0, 0)),
    #          (rear_surface,  (0, 0, thickness),     (0, 0, 0)),
    #          (edge,          (0, 0, thickness/2.0), (0, 0, 0))]
    #component = Component(surflist=surflist, material=schott["BK7"])
    component = Component(surflist=[(front_surface, (0, 0, 0),             (0, 0, 0))], material=schott["BK7"])
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

def create_rays(screen):
    """
    For now, just make a bunch of rays coming off of the screen.
    """
    fov = math.pi/2.0
    return [Ray(pos=screen.vision_ray_to_pixel(AngleVector(theta, phi)), dir=screen.direction) \
            for (theta, phi) in \
            [(0, 0), (0, fov), (math.pi/2.0, fov), (math.pi, fov), (3.0*math.pi/2.0, fov)]]

def main():
    #create the main app
    app = wx.PySimpleApp()

    #system assumptions
    FOV = 45.0
    screen_angle = math.pi / 8.0
    principal_eye_vector = Point3D(0.0, 0.0, -1.0)

    #create the components
    screen_location = Point3D(0, 40.0, -20.0)
    screen_direction = Point3D(0, -math.sin(screen_angle), -math.cos(screen_angle))
    screen_size = Point2D(25.0, 25.0)
    def pixel_distribution(vec):
        r = vec.phi / FOV
        return Point2D(r*math.cos(vec.theta), r*math.sin(vec.theta))
    screen = Screen(screen_location, screen_direction, screen_size, pixel_distribution)

    shell_distance = 60.0
    shell_radius = 80.0
    shell = create_shell(shell_distance, principal_eye_vector, shell_radius)
    detector = create_detector()
    raylist = create_rays(screen)

    #assemble them into the system
    #system = System(complist=[screen.create_component(), shell, detector], n=1)
    system = System(complist=[shell, detector], n=1)
    system.ray_add(raylist)

    #run the simulation
    system.propagate()
    glPlotFrame(system)
    spot_diagram(detector.ccd)
    app.MainLoop()

def tutorial():
    app = wx.PySimpleApp()
    shape=Circular(radius=20.)
    cohef = numpy.array([[0,0.01],[0.01,0.02]]).copy(order='C')
    S1=TaylorPoly(shape=shape, cohef=cohef, reflectivity=1.0)
    S2=Spherical(curvature=1/200., shape=Circular(radius=20.))
    S3=Cylinder(radius=20,length=6.997)
    surflist=[(S1, (0, 0, -5), (0, 0, 0)),
              (S2, (0, 0, 5), (0, math.pi, 0)),
              (S3,(0,0,.509),(0,0,0))]
    L1=Component(surflist=surflist, material=schott["BK7"])
    ccd=CCD(size=(100,100), transparent=False)
    S=System(complist=[(L1, (0, 0, 20), (0, 0, 0)),(ccd, (0, 0, 150), (0, 0, 0))], n=1)
    R=[Ray(pos=(0, 0, 0), dir=(0, 0, 1)), Ray(pos=(10, 0, 0), dir=(0, 0, 1)), Ray(pos=(-10, 0, 0), dir=(0, 0, 1)),Ray(pos=(0, 10, 0), dir=(0, 0, 1)), Ray(pos=(0, -10, 0), dir=(0, 0, 1)),]
    S.ray_add(R)
    S.propagate()
    glPlotFrame(S)
    spot_diagram(ccd)
    app.MainLoop()

if __name__ == '__main__':
    main()

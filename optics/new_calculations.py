
"""
An attempt to rewrite using patches, and not mixed in with a bunch of dead code
Goal is to be able to delete calculations.py when finished with this one
"""

import os
import sys
import time
import math
import itertools
import threading
import traceback
import string
import random
import pickle

import numpy
import scipy.integrate
import scipy.optimize

#this is the one thing that is allowed to import *
from optics.base import *
import optics.globals
import optics.utils
import optics.parallel
import optics.scale
import optics.taylor_poly
import optics.arc
import optics.arcplane

FORCE_FLAT_SCREEN = True
FOV = math.pi / 2.0

ORIGIN = Point3D(0.0, 0.0, 0.0)
PRINCIPAL_RAY = Point3D(0.0, 0.0, -1.0)
FORWARD = 1.0
BACKWARD = -1.0

def phi_to_pixel_size(phi, theta):
    """
    Controls how fast the screen shrinks at the edges
    """
    max_phi = FOV/2.0
    max_pixel_size = 0.080
    min_pixel_size = 0.005
    pixel_size_delta = max_pixel_size - min_pixel_size
    return max_pixel_size - pixel_size_delta * (phi / max_phi)

def create_rib_arcs(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_patch):
    """
    Creates a set of ribs and returns all of those arcs.
    Really just here to see basic shape and performance.
    """
    
    #temporary parameter. Defines whether we make the spine vertically or horizontally
    vertical_first = True
    
    #based on the fact that your pupil is approximately this big
    #basically defines how big the region is that we are trying to put in focus with a given patch
    light_radius = 3.0
    
    #calculated based on light radius. how far we should step between each arc basically
    angle_step = math.atan(light_radius, initial_shell_point[2])
    
    #create the spine halves
    if vertical_first:
        primary_arc_plane = ArcPlane(mu=0.0)
    else:
        primary_arc_plane = ArcPlane(rho=0.0)
    spines = [grow_axis(initial_shell_point, initial_screen_point, screen_normal, primary_arc_plane, direction, angle_step) for direction in (FORWARD, BACKWARD)]
    
    #starting from each of the end points, create some more arcs
    all_arcs = list()
    points = [(arc.start_point, arc.screen_point) for arc in all_arcs][1:] #trims one from the start so we don't duplicate the origin
    for shell_point, screen_point in points:
        all_arcs += grow_axis(shell_point, screen_point, screen_normal, arc_plane, FORWARD, angle_step)
        all_arcs += grow_axis(shell_point, screen_point, screen_normal, arc_plane, BACKWARD, angle_step)
    
    return all_arcs

def grow_axis(initial_shell_point, initial_screen_point, screen_normal, arc_plane, direction, angle_step):
    """
    Grows a set of arcs along the arc plane.
    Each arc is optimized to focus light to the starting point.
    Then we calculate where it happens to focus light for the ending point, and use that as the place for the next arc to focus.
    
    :param initial_shell_point: where the first arc start from
    :type  initial_shell_point: Point3D
    :param initial_screen_point: where to focus the first arc
    :type  initial_screen_point: Point3D
    :param arc_plane: all arcs will exist only in this plane
    :type  arc_plane: ArcPlane
    :param direction: Which direction to grow the arcs within the arc_plane
    :type  direction: float
    
    :returns: all of the arcs that were created, in order
    :rtype: list(Arc)
    """
    arcs = []
    angle = 0.0
    shell_point = arc_plane.world_to_local(initial_screen_point)
    prev_screen_point = arc_plane.world_to_local(initial_screen_point)
    focal_screen_point = arc_plane.world_to_local(initial_screen_point)
    transformed_screen_normal = normalize(arc_plane.world_to_local(screen_normal))
    while angle < FOV:
        angle += angle_step
        if arc_plane.rho == None:
            end_arc_plane = ArcPlane(rho=angle)
        else:
            end_arc_plane = ArcPlane(mu=angle)
        arc = optics.arc.grow_arc(shell_point, focal_screen_point, transformed_screen_normal, prev_screen_point, arc_plane, end_arc_plane)
        arcs.append(arc)
        shell_point = arc.end_point
        prev_screen_point = arc.screen_point
        focal_screen_point = get_focal_point(arc)
    return arcs
    
def get_focal_point(arc):
    """
    Figure out where light would be focused for this arc
    """
    if FORCE_FLAT_SCREEN:
        #just finds the average location of all of the places where the rays hit
        rays = blah
        screen_points = cast_rays_on_to_screen(rays, [arc])
        filtered_points = [p for p in screen_points if p != None]
        return sum(filtered_points) / len(filtered_points)
    else:
        #calculate based on the bundle of rays.
        raise NotImplementedError()
    
def rotate_90(point):
    return Point2D(-point[1], point[0])

def cast_rays_on_to_screen(rays, arcs, screen=None):
    """
    returns the locations on the screen
    """
    if FORCE_FLAT_SCREEN:
        arc = arcs[0]
        screen_line_start = arc.screen_point
        screen_line_end = rotate_90(arc.screen_normal) + arc.screen_point
        for ray in rays:
            intersection = arc.fast_arc_plane_intersection(ray)
            if intersection == None:
                screen_points.append(None)
            else:
                normal = arc.fast_normal(intersection)
                
                screen_points.append(blah)
        return screen_points
    else:
        #calculate based on the bundle of rays.
        raise NotImplementedError()
    
_H_ARC_NORMAL = get_arc_plane_normal(PRINCIPAL_RAY, True)
_V_ARC_NORMAL = get_arc_plane_normal(PRINCIPAL_RAY, False)
def get_angle_vec_from_point(point):
    return AngleVector(
        get_theta_from_point(PRINCIPAL_RAY, _H_ARC_NORMAL, _V_ARC_NORMAL, point),
        normalized_vector_angle(PRINCIPAL_RAY, normalize(point)))

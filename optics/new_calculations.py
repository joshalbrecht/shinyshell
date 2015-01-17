
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

def create_rib_arcs(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_arc):
    """
    Creates a set of ribs and returns all of those arcs.
    Really just here to see basic shape and performance.
    """
    
    #temporary parameter. Defines whether we make the spine vertically or horizontally
    vertical_first = True
    
    #calculated based on light radius. how far we should step between each arc basically
    angle_step = math.fabs(math.atan(optics.globals.LIGHT_RADIUS / initial_shell_point[2]))
    
    #create the spine halves
    if vertical_first:
        primary_arc_plane = optics.arcplane.ArcPlane(mu=0.0)
    else:
        primary_arc_plane = optics.arcplane.ArcPlane(rho=0.0)
    spines = [grow_axis(initial_shell_point, initial_screen_point, screen_normal, primary_arc_plane, direction, angle_step, on_new_arc) for direction in (FORWARD, BACKWARD)]
    
    #starting from each of the end points, create some more arcs
    all_arcs = spines[0] + spines[1]
    check_performance(all_arcs)
    points = [(arc.start_point, arc.screen_point) for arc in all_arcs][1:] #trims one from the start so we don't duplicate the origin
    for shell_point, screen_point in points:
        all_arcs += grow_axis(shell_point, screen_point, screen_normal, arc_plane, FORWARD, angle_step, on_new_arc)
        all_arcs += grow_axis(shell_point, screen_point, screen_normal, arc_plane, BACKWARD, angle_step, on_new_arc)
    
    return all_arcs

def check_performance(arcs):
    max_angle = FOV/2.0
    num_rays = 11
    ray_length = 1000.0
    #for angle in numpy.linspace(-max_angle, max_angle, 29):
    for angle in numpy.linspace(0.0, max_angle, 29):
        #make a bunch of rays
        normal = Point2D(math.cos(angle), math.sin(angle))
        rays = []
        for offset in numpy.linspace(-optics.globals.LIGHT_RADIUS, optics.globals.LIGHT_RADIUS, num_rays):
            delta = Point2D(0.0, offset)
            rays.append(Ray(delta, delta + normal * ray_length))
        #check them against all arcs (cause I'm lazy)
        intersections, screen_points = cast_rays_on_to_screen(rays, arcs)
        filtered_points = numpy.array([p for p in screen_points if p != None])
        #report size of focal point
        focal_point = sum(filtered_points) / len(filtered_points)
        distances = [numpy.linalg.norm(p-focal_point) for p in filtered_points]
        spot_error = sum(distances) / len(filtered_points)
        print("%.2f   %.7f" % (angle, spot_error))
    
def grow_axis(initial_shell_point, initial_screen_point, screen_normal, arc_plane, direction, angle_step, on_new_arc):
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
    shell_point = arc_plane.world_to_local(initial_shell_point)
    prev_screen_point = arc_plane.world_to_local(initial_screen_point)
    focal_screen_point = arc_plane.world_to_local(initial_screen_point)
    transformed_screen_normal = normalize(arc_plane.world_to_local(screen_normal))
    while math.fabs(angle) < FOV/2.0:
        angle += direction * angle_step
        if arc_plane.rho == None:
            end_arc_plane = optics.arcplane.ArcPlane(rho=angle)
        else:
            end_arc_plane = optics.arcplane.ArcPlane(mu=angle)
        arc = optics.arc.grow_arc(shell_point, focal_screen_point, transformed_screen_normal, prev_screen_point, arc_plane, end_arc_plane)
        arcs.append(arc)
        on_new_arc(arc)
        shell_point = arc.end_point
        prev_screen_point = arc.screen_point
        focal_screen_point = get_focal_point(arc)
    return arcs

def _generate_rays(arc, main_ray_vector):
    directions = get_spaced_points(arc.start_point, arc.end_point)
    rays = []
    for direction in directions:
        start_point = direction - main_ray_vector
        end_point = start_point + 2.0 * main_ray_vector
        rays.append(Ray(start_point, end_point))
    return rays
    
def get_focal_point(arc):
    """
    Figure out where light would be focused for this arc
    """
    if FORCE_FLAT_SCREEN:
        #just finds the average location of all of the places where the rays hit
        rays = _generate_rays(arc, arc.end_point)
        intersections, screen_points = cast_rays_on_to_screen(rays, [arc])
        filtered_points = numpy.array([p for p in screen_points if p != None])
        return sum(filtered_points) / len(filtered_points)
    else:
        #calculate based on the bundle of rays.
        raise NotImplementedError()

def cast_rays_on_to_screen(rays, arcs, screen=None):
    """
    returns the locations on the screen
    """
    if FORCE_FLAT_SCREEN:
        screen_line_start = arcs[0].screen_point
        screen_line_end = rotate_90(arcs[0].screen_normal) + arcs[0].screen_point
        intersections = []
        screen_points = []
        #arc.draw_rays(rays)
        for ray in rays:
            intersection = None
            for arc in arcs:
                intersection = arc.fast_arc_plane_intersection(ray)
                if intersection != None:
                    intersections.append(intersection)
                    normal = arc.fast_normal(intersection)
                    reverse_ray_direction = normalize(ray.start - ray.end)
                    midpoint = closestPointOnLine(reverse_ray_direction, Point2D(0.0, 0.0), normal)
                    reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
                    ray_to_screen_end = intersection + reflection_direction
                    screen_intersection = intersect_lines((intersection, ray_to_screen_end), (screen_line_start, screen_line_end))
                    screen_points.append(screen_intersection)
                    break
                if intersection == None:
                    screen_points.append(None)
                    intersections.append(None)
        return intersections, screen_points
    else:
        #calculate based on the bundle of rays.
        raise NotImplementedError()
    


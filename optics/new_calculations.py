
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

FORCE_FLAT_SCREEN = False
FOV = 70.0 * math.pi / 180.0

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

def create_patch(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_arc):
    max_angle = math.fabs(math.atan(optics.globals.LIGHT_RADIUS / initial_shell_point[2]))
    
    #create up rib
    primary_vertical_plane = optics.arcplane.ArcPlane(mu=0.0)
    ending_horizontal_plane = optics.arcplane.ArcPlane(rho=max_angle)
    left_side_arc = grow_arc(initial_shell_point, initial_screen_point, initial_screen_point, primary_vertical_plane, ending_horizontal_plane)
    on_new_arc(left_side_arc)
    
    #create right rib
    primary_horizontal_plane = optics.arcplane.ArcPlane(rho=0.0)
    ending_vertical_plane = optics.arcplane.ArcPlane(mu=max_angle)
    bottom_arc = grow_arc(initial_shell_point, initial_screen_point, initial_screen_point, primary_horizontal_plane, ending_vertical_plane)
    on_new_arc(bottom_arc)
    
    #grow to the right from the first vertical arc
    left_side_focal_point = left_side_arc.arc_plane.local_to_world(get_focal_point(left_side_arc))
    top_arc = grow_arc(left_side_arc.world_end_point,
                       left_side_focal_point,
                       left_side_focal_point,
                       ending_horizontal_plane, ending_vertical_plane)
    on_new_arc(top_arc)
    
    #grow up from the second horizontal arc
    #RESUME: focal point makes no sense
    bottom_focal_point = bottom_arc.arc_plane.local_to_world(get_focal_point(bottom_arc))
    right_side_arc = grow_arc(bottom_arc.world_end_point,
                       bottom_focal_point,
                       bottom_focal_point,
                       ending_vertical_plane, ending_horizontal_plane)
    on_new_arc(right_side_arc)
    
    #plot everything in 3D
    #measure the distance between the two end points
    return [bottom_arc, top_arc, left_side_arc, right_side_arc]


def create_rib_arcs(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_arc):
    """
    Creates a set of ribs and returns all ogrow_arc(initial_shell_point, initial_screen_point, prev_screen_point, arc_plane, end_arc_plane)f those arcs.
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
    #check_performance(all_arcs)
    #draw_things(all_arcs)
    points = [(arc.arc_plane.local_to_world(arc.shell_point), arc.arc_plane.local_to_world(arc.screen_point)) for arc in all_arcs][1:] #trims one from the start so we don't duplicate the origin
    for shell_point, screen_point in points:
        if vertical_first:
            arc_plane = optics.arcplane.ArcPlane(rho=math.atan(shell_point[1] / -shell_point[2]))
        else:
            arc_plane = optics.arcplane.ArcPlane(mu=math.atan(shell_point[0] / -shell_point[2]))
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
        spot_error = _calculate_performance(screen_points)
        print("%.2f   %.7f" % (angle, spot_error))
        
def _calculate_performance(screen_points):
    filtered_points = numpy.array([p for p in screen_points if p != None])
    focal_point = sum(filtered_points) / len(filtered_points)
    distances = [numpy.linalg.norm(p-focal_point) for p in filtered_points]
    spot_error = sum(distances) / len(filtered_points)
    return spot_error
        
def draw_things(arcs):
    import viewer.scene_objects
    max_angle = FOV/2.0
    num_rays = 11
    ray_length = 1000.0
    rays = []
    for angle in numpy.linspace(-max_angle, max_angle, 29):
    #for angle in numpy.linspace(0.0, max_angle, 29):
        #make a bunch of rays
        normal = Point2D(math.cos(angle), math.sin(angle))
        for offset in numpy.linspace(-optics.globals.LIGHT_RADIUS, optics.globals.LIGHT_RADIUS, num_rays):
            delta = Point2D(0.0, offset)
            ray = Ray(delta, delta + normal * ray_length)
            for arc in arcs:
                intersection, reflection_direction = arc.fast_arc_plane_reflection(ray)
                if intersection != None:
                    #arc._debug_plot_intersection(ray)
                    transform = arc.arc_plane.local_to_world
                    reflection_length = numpy.linalg.norm(arc.shell_point - arc.screen_point) * 1.05
                    rays.append(viewer.scene_objects.LightRay(transform(ray.start), transform(intersection)))
                    rays.append(viewer.scene_objects.LightRay(transform(intersection), transform(intersection + reflection_direction * reflection_length)))
                    break
    arcs[0].render_rays = rays
    
def grow_axis(initial_shell_point, initial_screen_point, screen_normal, arc_plane, direction, angle_step, on_new_arc):
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
    
def grow_arc(initial_shell_point, initial_screen_point, prev_screen_point, arc_plane, end_arc_plane):
    shell_point = arc_plane.world_to_local(initial_shell_point)
    prev_screen_point = arc_plane.world_to_local(prev_screen_point)
    focal_screen_point = arc_plane.world_to_local(initial_screen_point)
    #TODO: entirely remove screen normal
    transformed_screen_normal = Point2D(1.0, 0.0)
    return optics.arc.grow_arc(shell_point, focal_screen_point, transformed_screen_normal, prev_screen_point, arc_plane, end_arc_plane)

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
        rays = _generate_rays(arc, arc.end_point)
        assert len(rays) % 2 == 1, "need a central ray"
        central_ray_idx = len(rays) / 2
        central_ray = rays[central_ray_idx]
        other_rays = rays[:central_ray_idx] + rays[central_ray_idx+1:]
        
        central_intersection, central_reflection_direction = arc.fast_arc_plane_reflection(central_ray)
        if central_intersection != None:
            central_reflected_ray_end = central_intersection + central_reflection_direction
            total = Point2D(0.0, 0.0)
            num_intersections = 0
            for ray in other_rays:
                intersection, reflection_direction = arc.fast_arc_plane_reflection(ray)
                if intersection != None:
                    total += intersect_lines(
                        [central_intersection, central_reflected_ray_end],
                        [intersection, intersection + reflection_direction]
                    )
                    num_intersections += 1
            screen_point = total / num_intersections
            
            #debug: checking performance
            screen_line_start = screen_point
            screen_line_end = rotate_90(normalize(central_intersection - screen_point)) + screen_point
            intersections, screen_points = cast_rays_on_to_flat_screen(rays, [arc], screen_line_start, screen_line_end)
            spot_error = _calculate_performance(screen_points)
            print("%.7f" % (spot_error))
            
            return screen_point 

def cast_rays_on_to_flat_screen(rays, arcs, screen_line_start, screen_line_end):
    intersections = []
    screen_points = []
    #arc.draw_rays(rays)
    for ray in rays:
        intersection = None
        for arc in arcs:
            intersection, reflection_direction = arc.fast_arc_plane_reflection(ray)
            if intersection != None:
                ray_to_screen_end = intersection + reflection_direction
                screen_intersection = intersect_lines((intersection, ray_to_screen_end), (screen_line_start, screen_line_end))
                screen_points.append(screen_intersection)
                intersections.append(intersection)
                break
            if intersection == None:
                screen_points.append(None)
                intersections.append(None)
    return intersections, screen_points
    

def cast_rays_on_to_screen(rays, arcs, screen_line_start=None, screen_line_end=None):
    """
    returns the locations on the screen
    """
    if FORCE_FLAT_SCREEN:
        if screen_line_end == None:
            screen_line_start = arcs[0].screen_point
        if screen_line_end == None:
            screen_line_end = rotate_90(arcs[0].screen_normal) + arcs[0].screen_point
        return cast_rays_on_to_flat_screen(rays, arcs, screen_line_start, screen_line_end)
    else:
        #TODO: have to find some way to collide with this new screen...
        #maybe fit a poly to the screen points and collide with that
        #raise NotImplementedError()
        return [None] * len(rays), [None] * len(rays)
    
    


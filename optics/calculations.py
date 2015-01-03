
"""
A bunch of functions for creating surfaces, casting rays, etc
"""

import math

import numpy
import scipy.integrate
import scipy.optimize

#this is the one thing that is allowed to import *
from optics.base import *
import optics.globals

def create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=None):
    assert is_horizontal != None, "Must pass this parameter"
    
    #define a vector field for the surface normals of the shell.
    #They are completely constrained given the location of the pixel and the fact
    #that the reflecting ray must be at a particular angle        
    arc_plane_normal = get_arc_plane_normal(principal_ray, is_horizontal)
    desired_light_direction_off_screen_towards_eye = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    return create_arc_helper(shell_point, screen_point, light_radius, arc_plane_normal, desired_light_direction_off_screen_towards_eye)
    
#just a continuation of the above function. allows you to pass in the normals so that this can work in taylor poly space
def create_arc_helper(shell_point, screen_point, light_radius, arc_plane_normal, desired_light_direction_off_screen_towards_eye):

    def f(point, t):
        point_to_screen_vec = normalize(screen_point - point)
        surface_normal = normalize(point_to_screen_vec + desired_light_direction_off_screen_towards_eye)
        derivative = normalize(numpy.cross(surface_normal, arc_plane_normal))
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
        t_step = 0.05
        if optics.globals.LOW_QUALITY_MODE:
            t_step = 0.5
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
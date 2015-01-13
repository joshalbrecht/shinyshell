
"""
A bunch of functions for creating surfaces, casting rays, etc
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

FOV = math.pi / 3.0

def phi_to_pixel_size(phi, theta):
    """
    Controls how fast the screen shrinks at the edges
    """
    max_phi = FOV/2.0
    max_pixel_size = 0.080
    min_pixel_size = 0.005
    pixel_size_delta = max_pixel_size - min_pixel_size
    return max_pixel_size - pixel_size_delta * (phi / max_phi)

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
    
    t_step = 0.1
    max_t = 1.1 * light_radius
    t_values = numpy.arange(0.0, max_t, t_step)

    #use the vector field to define the exact shape of the surface (first half)
    half_arc = scipy.integrate.odeint(f, shell_point, t_values)
    
    #do the other half as well
    def g(point, t):
        return -1.0 * f(point, t)
    
    #combine them
    every_nth = 10
    return numpy.concatenate((scipy.integrate.odeint(g, shell_point, t_values)[::every_nth][1:][::-1], half_arc[::every_nth]))

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m

#Note--might seem a little bizarre that we are transforming everything outside of PolyShell even though the details of its inner workings should be concealed
#but it's for efficiency reasons--matrix multiplying a bajillion points into the correct space is going to be way slower than just
#making them in the correct coordinate system in the first place
#really, should probably hide create_arc inside of PolyScale, but it's used elsewhere, so leaving it out for now
def make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec, poly_order, screen_normal):
    """
    returns a non-trimmed scale patch based on the point (where the shell should be centered)
    angle_vec is passed in for our convenience, even though it is duplicate information (given the shell_point)
    
    note: poly_order=4 is very high quality. decrease to 2 or 3 for polynomials that are not as good at approximating, but much faster
    """
    
    #taylor polys like to live in f(x,y) -> z
    #so build up the transformation so that the average of the shell -> screen vector and desired light vector is the z axis
    #eg, so the 0,0,0 surface normal is the z axis
    shell_to_screen_normal = normalize(screen_point - shell_point)
    desired_light_dir = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    z_axis_world_dir = normalize(desired_light_dir + shell_to_screen_normal)
    world_to_local_translation = -1.0 * shell_point
    world_to_local_rotation = numpy.zeros((3, 3))
    optics.rotation_matrix.R_2vect(world_to_local_rotation, z_axis_world_dir, Point3D(0.0, 0.0, 1.0))
    
    def translate_to_local(p):
        return world_to_local_rotation.dot(p + world_to_local_translation)
    
    #convert everything into local coordinates
    transformed_light_dir = world_to_local_rotation.dot(desired_light_dir)
    h_arc_plane_normal = get_arc_plane_normal(principal_ray, True)
    v_arc_plane_normal = get_arc_plane_normal(principal_ray, False)
    transformed_screen_point = translate_to_local(screen_point)
    transformed_shell_point = Point3D(0.0, 0.0, 0.0)
    
    #actually go calculate the points that we want to use to fit our polynomial
    spine = optics.calculations.create_arc_helper(transformed_shell_point, transformed_screen_point, light_radius, v_arc_plane_normal, transformed_light_dir)
    ribs = []
    for point in spine:
        rib = optics.calculations.create_arc_helper(point, transformed_screen_point, light_radius, h_arc_plane_normal, transformed_light_dir)
        ribs.append(numpy.array(rib))
        
    points = numpy.vstack(ribs)
    #fit the polynomial to the points:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    coefficients = polyfit2d(x, y, z, order=poly_order)
    order = int(numpy.sqrt(len(coefficients)))
    cohef = []
    for i in range(0, order):
        cohef.append(coefficients[i*order:(i+1)*order])
    cohef = numpy.array(cohef).copy(order='C')
    poly = optics.taylor_poly.TaylorPoly(cohef=cohef.T, domain_radius=light_radius, domain_point=translate_to_local(screen_point))
    
    scale = optics.scale.PolyScale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        poly=poly,
        world_to_local_rotation=world_to_local_rotation,
        world_to_local_translation=world_to_local_translation,
        domain_cylinder_radius=light_radius,
        screen_normal=screen_normal
    )
    return scale

#TODO: this arc creation should probably be parallelized. Trivially parallelizable
#TODO: need a reasonable method for looking up the closest scales
def create_averaged_surface(scales):
    """
    Working along x and y arcs, get the distance by intersecting with the relevant scales
    """
    
    for y in range():
        for x in range():
            code

def even_newer_calculate_error(scale, reference_scale, best_error_so_far):
    """
    Measure the optical performance of a few points that are in between the two scales
    """

    #TODO: unhack reference_scale.screen_normal
    screen_plane = Plane(reference_scale._pixel_point, reference_scale.screen_normal)
    
    #make a vector between the two scales and split into a few different pieces
    num_primary_rays = 6
    end_point_vector = scale.shell_point - reference_scale.shell_point
    end_point_vector_length = numpy.linalg.norm(end_point_vector)
    end_point_normal = end_point_vector / end_point_vector_length
    end_point_distances = numpy.linspace(0, end_point_vector_length, num_primary_rays)
    end_points = [dist * end_point_normal + reference_scale.shell_point for dist in end_point_distances]
    
    #calculate reflections for all bundles of rays centered around the end points
    worst_error_this_iteration = 0.0
    beams_per_ray = 10
    #TODO: this should almost certainly be light radius...
    pupil_radius = 3.0
    for end_point in end_points:
        #figure out where that ray would end up on the screen. that is the pixel point for this bundle of rays
        primary_ray = Ray(Point3D(0.0, 0.0, 0.0), end_point * 2.0)
        primary_shell_collision, primary_screen_collision = optics.scale.get_best_shell_and_screen_point_from_ray(reference_scale, scale, primary_ray, screen_plane)
        
        #make the bundle of rays, and reflect them all on to the screen
        ray_end = primary_shell_collision
        #cumulative_distance = 0.0
        #num_collisions = 0
        for y in numpy.linspace(-pupil_radius, pupil_radius, num=beams_per_ray):
            delta = Point3D(0, y, 0)
            ray = Ray(delta, ray_end + delta)
            shell_collision, screen_collision = optics.scale.get_best_shell_and_screen_point_from_ray(reference_scale, scale, ray, screen_plane)
            if shell_collision != None:
                #self._rays.append(LightRay(delta, shell_collision))
                #self._rays.append(LightRay(shell_collision, screen_collision))
                #cumulative_distance += numpy.linalg.norm(primary_screen_collision - screen_collision)
                #num_collisions += 1
                
                #update the error
                dist = numpy.linalg.norm(primary_screen_collision - screen_collision)
                if dist > worst_error_this_iteration:
                    worst_error_this_iteration = dist
                    #if the distance is worse than the best error so far, return this error
                    if dist > best_error_so_far:
                        return dist
                    
        #TODO: calculate MTF instead
        #print(cumulative_distance / num_collisions)
    return worst_error_this_iteration
        
#TODO: this method for calculating scale error is not EXTREMELY accurate.
#It's not bad, it's just that we're taking a relatively small number of samples,
#and not necessarily the ones closest to the edge of the scale
#could easily augment this by taking more samples near the edge, even if they didn't impact
#the other constraint about which scale was closer
def new_calculate_error(scale, reference_scale, best_error_so_far):
    """
    In theory, the error we are trying to calculate is the maximal distance between the two scales (from any two points
    that are actually in domain).
    
    In practice, we are exploiting a few known facts about the reference scale (like that it is more central and reasonably close
    and symmetrical and partially overlapping)
    
    Note that we also have a constraint about the way in which the reference and scale should overlap. Basically, we do not want
    discontinuities, so we ensure that as soon as the new scale is closer to the eye than the reference, it has to stay that way.
    Thus as you work outward from the center, there will not usually be any points where the scales are overlapping in a way
    that makes discontinuities.
    
    Given all of that, our general approach is to look for the scale points that are farthest from the scale -> reference scale line.
    These points define the maximal angle that we need to consider. For a few angles inbetween those, simply look at the intersections
    of that plane with both scales, ensuring that the above constraint is satisfied, and returning the largest distance
    """
    #find outermost points
    scale_points = scale.points()
    best_dist_sq = 0.0
    best_point = None
    for point in scale_points:
        dist_sq = distToLineSquared(point, scale.shell_point, reference_scale.shell_point)
        if dist_sq > best_dist_sq:
            best_dist_sq = dist_sq
            best_point = point
            
    #use that to define a few end points for rays that we will walk along, calculating distance
    best_dist = math.sqrt(best_dist_sq)
    nearest_point = closestPointOnLine(best_point, scale.shell_point, reference_scale.shell_point)
    ray_end_normal = normalize(best_point - nearest_point)
    #TODO: do we ever need more than this? Do we even need this?
    num_rays = 3
    distances = numpy.linspace(-best_dist, best_dist, num_rays+2)[1:-1]
    #TODO: this will not work if we make the shell vary out of the obvious axis
    if optics.globals.QUALITY_MODE == optics.globals.ULTRA_LOW_QUALITY_MODE:
        distances = [0.0]
    
    end_points = [nearest_point + dist*ray_end_normal for dist in distances]
    
    #TODO: constants below are rather arbitrary, and unlinked to anything else. should probably be though through a bit more
    
    #for each ray, walk along calculating distances between the two scales at various points
    #note that rays have to start really far away
    ray_dist = 1000.0
    start_point = ray_dist * -1.0 * normalize(nearest_point - reference_scale.shell_point) + scale.shell_point
    #we want to start casting somewhere near where the scale could possibly overlap with the reference scale
    fudge_factor = 1.05 #start a little ways away from the shell. who knows what would happen at the edge
    starting_ray_distance = ray_dist - scale._poly.get_radius() * fudge_factor
    #and we have to end within a reasonable amount (no need to go much farther than 2x the radius)
    ending_ray_distance = ray_dist + scale._poly.get_radius() * fudge_factor
    num_intersections_per_ray = 10 #sure, whatever, seems fine?
    ray_distances = numpy.linspace(starting_ray_distance, ending_ray_distance, num_intersections_per_ray+2)[1:-1]
    ray_start = Point3D(0.0, 0.0, 0.0) #all rays start from the eye
    worst_error_this_iteration = 0.0
    for end_point in end_points:
        ray_normal = normalize(end_point - start_point)
        hit_reference_already = False
        hit_scale_first_already = False
        for dist in ray_distances:
            #actually perform the intersections
            ray_end = 2.0 * (dist * ray_normal + start_point)
            reference_intersection = reference_scale.intersection(ray_start, ray_end)
            
            #if we've ever hit the reference scale during this ray, and we didn't this time, break
            if reference_intersection == None:
                if hit_reference_already:
                    break
            else:
                hit_reference_already = True
            
            #if we didn't hit both scales, continue
            scale_intersection = scale.intersection(ray_start, ray_end)
            if scale_intersection == None or reference_intersection == None:
                continue
            
            #if our assumption is violated, return infinite error
            reference_dist_sq = numpy.dot(reference_intersection, reference_intersection)
            scale_dist_sq = numpy.dot(scale_intersection, scale_intersection)
            if scale_dist_sq > reference_dist_sq:
                if hit_scale_first_already:
                    return float("inf")
            else:
                hit_scale_first_already = True

            #update the error
            dist = math.fabs(math.sqrt(reference_dist_sq) - math.sqrt(scale_dist_sq))
            if dist > worst_error_this_iteration:
                worst_error_this_iteration = dist
                #if the distance is worse than the best error so far, return this error
                if dist > best_error_so_far:
                    return dist
                
    #NOTE: neither of these seems that important right now because this doesn't seem like it is the bottleneck
    
    #OPT: it might then be possible to look at my optimizations (calculate z instead of doing collisions)
    #which could dramatically speed up the entire operation.
    
    #OPT: I wonder if converting this to cython could help at all...
    #seems like it. let's save for the future: http://docs.cython.org/src/tutorial/numpy.html
    #also before doing that, worth double checking in the profiler that the calculate_error function is the most time intensive
    
    return worst_error_this_iteration
    
    
def calculate_error(scale, reference_scale, best_error_so_far):
    """
    I guess shoot rays all over the scale (from the pixel location), and see which also hit the reference scale, and get the distance
    least squares? or just sum all of it? I wonder why people use least squares all the time...
    note: will have to be average error per sample point, since different shells will have different number of sample points
    question is just whether to average the squares, or regularize them
    """
    start = reference_scale.pixel_point
    dist = 0.0
    worst_error_this_iteration = 0.0
    num_hits = 0
    points = reference_scale.points()
    for point in points:
        end = 2.0 * (point - start) + start
        intersection_point = scale.intersection(start, end)
        if intersection_point != None:
            num_hits += 1
            #print numpy.linalg.norm(intersection_point - point)
            dist = numpy.linalg.norm(intersection_point - point)
            if dist > worst_error_this_iteration:
                worst_error_this_iteration = dist
                if dist > best_error_so_far:
                    return dist
            #delta = intersection_point - point
            #dist += delta.dot(delta)
    #average_error = dist / num_hits
    #print num_hits
    #return average_error
    return worst_error_this_iteration

def _get_scale_and_error_at_distance(distance, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec, poly_order, best_error_so_far):
    shell_point = distance * angle_normal
    scale = make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec, poly_order)
    error = max([even_newer_calculate_error(scale, reference_scale, best_error_so_far) for reference_scale in reference_scales])
    scale.shell_distance_error = error
    return scale, error

def find_scale_and_error_at_best_distance(reference_scales, principal_ray, screen_point, light_radius, angle_vec, best_error_this_iteration):
    """
    iteratively find the best distance that this scale can be away from the reference scales
    """
    
    start_time = time.time()
    
    #seems pretty arbitrary, but honestly at that point the gains here are pretty marginal
    num_iterations = 20
    if optics.globals.QUALITY_MODE == optics.globals.ULTRA_LOW_QUALITY_MODE:
        tolerance = 0.5
    elif optics.globals.QUALITY_MODE == optics.globals.LOW_QUALITY_MODE:
        tolerance = 0.1
    else:
        tolerance = 0.001
        
    poly_order = optics.globals.POLY_ORDER
    
    angle_normal = angle_vector_to_vector(angle_vec, principal_ray)
    reference_distance = numpy.linalg.norm(reference_scales[0].shell_point)
    
    lower_bound_dist = reference_distance - light_radius
    upper_bound_dist = reference_distance + light_radius

    scales = {}
    def f(x):
        scale, error = _get_scale_and_error_at_distance(x, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec, poly_order, best_error_this_iteration[0])
        scales[x] = (scale, error)
        if error < best_error_this_iteration[0]:
            best_error_this_iteration[0] = error
        return error
    #best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=0.001, full_output=True, disp=3)
    best_value = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    
    print("Time: %s" % (time.time() - start_time))
    
    return scales[best_value]

#TODO: do we need more than just a cross here?
def _generate_rays(end_point, light_radius, num_rays=11):
    """make a bundle of rays"""
    rays = []
    ray_end = end_point * 2.0
    for v in numpy.linspace(-light_radius, light_radius, num=num_rays):
        delta = Point3D(0, v, 0)
        rays.append(Ray(delta, ray_end + delta))
        delta = Point3D(v, 0, 0)
        rays.append(Ray(delta, ray_end + delta))
    return rays

def new_explore_direction(screen_normal, prev_scale, nearby_scales, principal_ray, light_radius, shell_growth_normal):
    """
    shell_growth_normal is roughly the direction that we want grow. Basically, the normal between the previous two scales.
    """
    #walk along a bunch of different arcs from -light_radius (out to the side) to +light_radius that travel in the growth direction
    #each arc needs to have at least one point (find the farthest point, skip domain trimming if necessary)
    #these arcs are the starting point for further growth below
    step_size = 0.1
    num_arcs = 1 + 2 * (int(light_radius / step_size) + 1)
    arc_offsets = numpy.linspace(-light_radius, light_radius, num_arcs)
    prev_shell_normal = normalize(prev_scale.pixel_point - prev_scale.shell_point)
    arc_offset_normal = numpy.cross(shell_growth_normal, prev_shell_normal)
    prev_scale_arcs = []
    for offset in arc_offsets:
        ray_start = offset * arc_offset_normal + prev_scale.shell_point
        final_arc_start = None
        final_arc = []
        for scale in nearby_scales:
            arc_start, arc = scale._get_arc_and_start(ray_start, shell_growth_normal, step_size)
            if final_arc_start == None:
                final_arc_start = arc_start
            if len(arc) > 0:
                final_arc_start = arc_start
                final_arc += arc
                ray_start = arc_start
        prev_scale_arcs.append((final_arc_start, final_arc))
    
    #grab the primary growth ray (from the center of the previous scale in the direction we want to grow)
    #cast a bunch of light rays at that point and reflect them from the shell on to the screen. the place where they focus will be our new screen point
    shell_point, primary_arc = prev_scale_arcs[num_arcs / 2]
    rays = _generate_rays(shell_point, light_radius)
    screen_plane = Plane(prev_scale.pixel_point, screen_normal)
    screen_points = prev_scale.get_screen_points(rays, screen_plane)
    assert len(screen_points) > 0
    focused_screen_point = sum(screen_points) / len(screen_points)
    
    #calculate angle_vector from shell_point
    h_arc_normal = get_arc_plane_normal(principal_ray, True)
    v_arc_normal = get_arc_plane_normal(principal_ray, False)
    angle_vec = AngleVector(get_theta_from_point(principal_ray, h_arc_normal, v_arc_normal, shell_point), normalized_vector_angle(principal_ray, normalize(shell_point)))
    
    #constrain screen_point to actually be along the growth vector
    screen_growth_normal = normalize(numpy.cross(arc_offset_normal, screen_normal))
    screen_point = closestPointOnLine(focused_screen_point, prev_scale._pixel_point, prev_scale._pixel_point + screen_growth_normal)
    
    ##TODO: maybe switch to this and use 3D bundle of rays?
    #screen_point = focused_screen_point
    #screen_growth_normal = normalize(screen_point - prev_scale._pixel_point)
    
    #cap the movement along the screen based on phi
    #max pixel size, ppd spec, and angular delta tells us exactly how much we can move
    pixel_size = phi_to_pixel_size((angle_vec.phi+prev_scale.angle_vec.phi)/2.0, (angle_vec.theta+prev_scale.angle_vec.theta)/2.0)
    pixels_per_degree = 30.0
    pixels_per_radian = pixels_per_degree * 180.0 / math.pi
    angular_delta = normalized_vector_angle(normalize(shell_point), normalize(prev_scale._shell_point))
    max_screen_distance = pixel_size * angular_delta * pixels_per_radian
    
    #if you move farther than that, the no, screen point gets truncated back towards the previous scale
    inter_pixel_distance = numpy.linalg.norm(screen_point - prev_scale._pixel_point)
    if inter_pixel_distance > max_screen_distance:
        screen_point = screen_growth_normal * max_screen_distance + prev_scale._pixel_point
        print("Trimming from %.4f to %.4f" % (inter_pixel_distance, max_screen_distance))

    #polyfit to make the taylor poly and return that new scale
    scale = new_make_scale(principal_ray, shell_point, screen_point, light_radius, optics.globals.POLY_ORDER, prev_scale_arcs, arc_offset_normal, step_size, screen_normal, angle_vec)
    return scale
    
def new_make_scale(principal_ray, shell_point, screen_point, light_radius, poly_order, prev_arcs, arc_plane_normal, step_size, screen_normal, angle_vec):
    """
    returns a non-trimmed scale patch based on the point (where the shell should be centered)
    
    note: poly_order=4 is very high quality. decrease to 2 or 3 for polynomials that are not as good at approximating, but much faster
    """
    
    #taylor polys like to live in f(x,y) -> z
    #so build up the transformation so that the average of the shell -> screen vector and desired light vector is the z axis
    #eg, so the 0,0,0 surface normal is the z axis
    shell_to_screen_normal = normalize(screen_point - shell_point)
    desired_light_dir = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    z_axis_world_dir = normalize(desired_light_dir + shell_to_screen_normal)
    world_to_local_translation = -1.0 * shell_point
    world_to_local_rotation = numpy.zeros((3, 3))
    optics.rotation_matrix.R_2vect(world_to_local_rotation, z_axis_world_dir, Point3D(0.0, 0.0, 1.0))
    
    def translate_to_local(p):
        return world_to_local_rotation.dot(p + world_to_local_translation)
    
    #convert everything into local coordinates
    transformed_light_dir = world_to_local_rotation.dot(desired_light_dir)
    #h_arc_plane_normal = get_arc_plane_normal(principal_ray, True)
    #v_arc_plane_normal = get_arc_plane_normal(principal_ray, False)
    #TODO: this feels a bit sketch-mode. is this right at all?
    transformed_arc_plane_normal = world_to_local_rotation.dot(arc_plane_normal)
    transformed_screen_point = translate_to_local(screen_point)
    #transformed_shell_point = Point3D(0.0, 0.0, 0.0)
    
    domain_sq_radius = light_radius * light_radius
    domain_point = normalize(transformed_screen_point)
    def in_domain(p):
        delta = p - (p.dot(domain_point) * domain_point)
        return delta.dot(delta) < domain_sq_radius
    
    #actually go calculate the points that we want to use to fit our polynomial
    #use a large upper t bound to make sure we make it far enough
    #TODO: 3.0 is totally arbitrary...
    max_t = 3.0 * light_radius
    t_values = numpy.arange(0.0, max_t, step_size)
    every_nth = 10
    #define a vector field to generate the exact shape of the surface
    new_arcs = []
    normer = numpy.linalg.norm
    for start_point, prev_arc in prev_arcs:
        def f(point, t):
            #actual code:
            #point_to_screen_vec = normalize(transformed_screen_point - point)
            #surface_normal = normalize(point_to_screen_vec + transformed_light_dir)
            #derivative = normalize(numpy.cross(surface_normal, arc_plane_normal))
            #return derivative
            
            #optimizing, should be equivalent:
            point_to_screen_vec = transformed_screen_point - point
            surface_vec = point_to_screen_vec / normer(point_to_screen_vec) + transformed_light_dir
            derivative = numpy.cross(surface_vec, arc_plane_normal)
            return derivative / normer(derivative)
        new_arc = scipy.integrate.odeint(f, translate_to_local(start_point), t_values)[::every_nth]
        #take every nth (appropriately) from old arc
        filtered_prev_arc = prev_arc[::-1][::every_nth][1:][::-1]
        #transform them to local
        transformed_prev_arc = [translate_to_local(p) for p in filtered_prev_arc]
        full_arc = list(transformed_prev_arc) + list(new_arc)
        #filter out any not in the new domain
        final_arc = [p for p in full_arc if in_domain(p)]
        if len(final_arc) > 0:
            new_arcs.append(final_arc)
        
    points = numpy.vstack(new_arcs)
    
    ##for debugging: export points so I can see wtf is happening with the weird ones
    #with open("%s_%s.points" % (angle_vec.theta, angle_vec.phi), 'wb') as outfile:
    #    outfile.write('\n'.join([str(p) for p in points]))
    
    #fit the polynomial to the points:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    coefficients = polyfit2d(x, y, z, order=poly_order)
    order = int(numpy.sqrt(len(coefficients)))
    cohef = []
    for i in range(0, order):
        cohef.append(coefficients[i*order:(i+1)*order])
    cohef = numpy.array(cohef).copy(order='C')
    poly = optics.taylor_poly.TaylorPoly(cohef=cohef.T, domain_radius=light_radius, domain_point=translate_to_local(screen_point))
    
    scale = optics.scale.PolyScale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        poly=poly,
        world_to_local_rotation=world_to_local_rotation,
        world_to_local_translation=world_to_local_translation,
        domain_cylinder_radius=light_radius,
        screen_normal=screen_normal
    )
    return scale

def explore_direction(optimization_normal, lower_bound, upper_bound, prev_scale, principal_ray, light_radius, angle_vec):
    num_iterations = 20
    if optics.globals.QUALITY_MODE == optics.globals.ULTRA_LOW_QUALITY_MODE:
        tolerance = 0.5
    elif optics.globals.QUALITY_MODE == optics.globals.LOW_QUALITY_MODE:
        tolerance = 0.1
    else:
        tolerance = 0.001
    
    results = {}
    best_error_this_iteration = [float("inf")]
    def f(x):
        pixel_point = prev_scale.pixel_point + x * optimization_normal
        #optics.utils.profile_line('find_scale_and_error_at_best_distance([prev_scale], principal_ray, pixel_point, light_radius, angle_vec, best_error_this_iteration)', globals(), locals())
        scale, error = find_scale_and_error_at_best_distance([prev_scale], principal_ray, pixel_point, light_radius, angle_vec, best_error_this_iteration)
        results[x] = (scale, error)
        return error
    best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound, upper_bound, maxfun=num_iterations, xtol=tolerance, full_output=True, disp=3)
    return results[best_value]

#def optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec):
#    
#    #optics.utils.profile_line('explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)', globals(), locals())
#    
#    approximately_correct_scale, decent_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
#    #print("Decent error: " + str(decent_error))
#    
#    #after that, simply find the point along that line (from the shell to that pixel) that is closest to the previous pixel
#    #(since we don't want the screen to get any bigger than it has to)
#    #and make the shell there
#    #TODO: will have to look at how the optimization curves look for surfaces where we are optimizing against 3 surfaces...
#    #might have to do another call to "explore_direction" to get the absolute best performance
#    best_screen_point = closestPointOnLine(prev_scale.pixel_point, approximately_correct_scale.pixel_point, approximately_correct_scale.shell_point)
#    #best_scale, error_for_best_scale = find_scale_and_error_at_best_distance([prev_scale], principal_ray, best_screen_point, light_radius, angle_vec)
#    #print("'best' error: " + str(error_for_best_scale))
#    
#    #doing another crawl along the line because why not
#    optimization_normal = normalize(best_screen_point - prev_scale.pixel_point)
#    final_scale, final_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
#    #print("Final error: " + str(final_error))
#    #scales.append(final_scale)
#    return final_scale, final_error
    
def create_screen_mesh(ordered_scales):
    arc = [scale.pixel_point for scale in ordered_scales]
    left_arc = [p + Point3D(-1.0, 0.0, 0.0) for p in arc]
    right_arc = [p + Point3D(1.0, 0.0, 0.0) for p in arc]
    return optics.mesh.Mesh(mesh=optics.mesh.mesh_from_arcs([right_arc, arc, left_arc]))

#TODO: can probably generalize this into a method that works with the whole 3D mesh. basically meant to take the most central collision
def _get_best_shell_and_screen_point_from_ray(prev_scale, scale, ray, screen_plane):
    """
    If we've collided with the previous scale, use that, otherwise, use collision with the next scale
    """
    shell, screen = prev_scale._get_shell_and_screen_point_from_ray(ray, screen_plane)
    if shell != None:
        return shell, screen
    
    shell, screen = scale._get_shell_and_screen_point_from_ray(ray, screen_plane)
    if shell != None:
        return shell, screen
    
    return None, None

def evaluate_scale(scale, prev_scale, light_radius):
    
    #make a vector between the two scales and split into a few different pieces
    num_primary_rays = 7
    end_point_vector = scale.shell_point - prev_scale.shell_point
    end_point_vector_length = numpy.linalg.norm(end_point_vector)
    end_point_normal = end_point_vector / end_point_vector_length
    end_point_distances = numpy.linspace(0, end_point_vector_length, num_primary_rays)
    end_points = [dist * end_point_normal + prev_scale.shell_point for dist in end_point_distances]
    
    #calculate reflections for all bundles of rays centered around the end points
    rays_per_bundle = 11
    rays_to_render = []
    pixel_errors = []
    screen_plane = Plane(prev_scale._pixel_point, prev_scale.screen_normal)
    for end_point in end_points:
        #figure out where that ray would end up on the screen. that is the pixel point for this bundle of rays
        ray_end = end_point * 2.0
        primary_ray = Ray(Point3D(0.0, 0.0, 0.0), end_point * 2.0)
        primary_shell_collision, primary_screen_collision = _get_best_shell_and_screen_point_from_ray(prev_scale, scale, primary_ray, screen_plane)
        
        #make the bundle of rays, and reflect them all on to the screen
        cumulative_distance = 0.0
        num_collisions = 0
        for y in numpy.linspace(-light_radius, light_radius, num=rays_per_bundle):
            delta = Point3D(0, y, 0)
            ray = Ray(delta, ray_end + delta)
            shell_collision, screen_collision = _get_best_shell_and_screen_point_from_ray(prev_scale, scale, ray, screen_plane)
            if shell_collision != None:
                rays_to_render.append(LightRay(delta, shell_collision))
                rays_to_render.append(LightRay(shell_collision, screen_collision))
                cumulative_distance += numpy.linalg.norm(primary_screen_collision - screen_collision)
                num_collisions += 1
        #TODO: calculate MTF instead
        #if num_collisions == 0:
        #    pixel_errors.append(None)
        #else:
        #    pixel_errors.append(cumulative_distance / num_collisions)
        pixel_errors.append(cumulative_distance / num_collisions)
    
    scale.focal_error = sum(pixel_errors) / len(pixel_errors)
    #TODO: find a way to enable the rendering from only a few scales instead of all of them
    #scale._rays = rays_to_render
    
    return pixel_errors
    
def create_surface_via_scales(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_scale):
    """
    Imagine a bunch of fish scales. Each represent a section of the shell, focused correctly for one pixel (eg, producing
    parallel rays heading towards the eye). By making a bunch of these, and adjusting the pixel locations so that they all line up,
    we should be able to make a surface that works well.
    
    Basic algorithm is a greedy one. Starting from the center scale, work outwards. Creates a hexagonal sort of mesh of these scales.
    Ask Josh for more details.
    """
    
    #based on the fact that your pupil is approximately this big
    #basically defines how big the region is that we are trying to put in focus with a given scale
    light_radius = 3.0
    ##per whole the screen. So 90 steps for a 90 degree total FOV would be one step per degree
    #total_phi_steps = 90

    ##calculated:
    #total_vertical_resolution = 2000
    #min_pixel_spot_size = 0.005
    #min_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * min_pixel_spot_size
    #max_pixel_spot_size = 0.015
    #max_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * max_pixel_spot_size
    
    #create the first scale
    center_scale = make_scale(principal_ray, initial_shell_point, initial_screen_point, light_radius, AngleVector(0.0, 0.0), optics.globals.POLY_ORDER, screen_normal)
    #center_scale.shell_distance_error = 0.0
    #center_scale.screen_normal = screen_normal
    #scales = [center_scale]
    
    on_new_scale(center_scale)

    ##create another scale right above it for debugging the error function
    ##shell_point = initial_shell_point + Point3D(0.0, 3.0, -1.0)#Point3D(0.0, 3.0, -1.0)
    #shell_point = initial_shell_point + Point3D(0.0, 3.0, -1.0)
    #angle_vec = AngleVector(math.pi/2.0, normalized_vector_angle(principal_ray, normalize(shell_point)))
    #other_scale = make_scale(principal_ray, shell_point, initial_screen_point+Point3D(0.0, -min_pixel_spot_size, min_pixel_spot_size), light_radius, angle_vec, optics.globals.POLY_ORDER)
    ##other_scale = make_scale(principal_ray, shell_point, initial_screen_point+Point3D(0.0, -200.0, 0.0), light_radius, angle_vec, optics.globals.POLY_ORDER)
    #ordered_scales = [center_scale, other_scale]
    #
    #start_time = time.time()
    #old_error = calculate_error(other_scale, center_scale, float("inf"))
    #end_time = time.time()
    #print("old: %s error in %s" % (old_error, end_time - start_time))
    #
    #start_time = time.time()
    #new_error = even_newer_calculate_error(other_scale, center_scale, float("inf"))
    #end_time = time.time()
    #print("new: %s error in %s" % (new_error, end_time - start_time))
    
    #other_scale, error = find_scale_and_error_at_best_distance([center_scale], principal_ray,
    #    #initial_screen_point+Point3D(0.0, -10.0, 10.0), light_radius, angle_vec)
    #    initial_screen_point+Point3D(0.0, 0.0, 0.0), light_radius, angle_vec)
    #print error
    
    #wheeee
    #now let's make a grid of different pixel locations, and how those impact the final error
    #scales = [center_scale, other_scale]
    #scales = [center_scale]
    
    ##make a 5x5 grid, centered on the previous screen location, and with +/- reasonable spacing * 2 in either direction
    #spacing = (max_spacing + min_spacing) / 2.0
    #grid_size = 9
    #plot_x, plot_y = numpy.meshgrid(
    #    numpy.linspace(initial_screen_point[2] - 5.0, initial_screen_point[2], grid_size),
    #    numpy.linspace(initial_screen_point[1], initial_screen_point[1] + 5.0, grid_size))
    #error_values = numpy.zeros((grid_size,grid_size))
    #for i in range(0, grid_size):
    #    for j in range(0, grid_size):
    #        z = plot_x[i][j]
    #        y = plot_y[i][j]
    #        other_scale, error = find_scale_and_error_at_best_distance([center_scale], principal_ray,
    #            #TODO: has a 0 in there, which will not generalize
    #            Point3D(0.0, y, z), light_radius, angle_vec)
    #        print error
    #        error_values[i][j] = error
    #plot_error(plot_x, plot_y, error_values)
    
    #ok, new approach to actually optimizing the next shell:
    #simply walk along the direction orthogonal to the last pixel -> shell vector in the current plane
    #and find the location with the minimal error
    
    #lower_bound = 0.0
    ##NOTE: is a hack / guestimate
    #upper_bound = 2.0 * light_radius
    ##upper_bound = max_spacing
        
    #phi_step = 0.05
    final_phi = 0.0001#FOV/6.0#FOV/2.0
    
    ##this is side to side motion
    #lateral_normal = normalize(numpy.cross(principal_ray, screen_normal))
    ##this defines the first arc that we are making (a vertical line along the middle)
    #optimization_normal = -1.0 * normalize(numpy.cross(lateral_normal, screen_normal))
    
    #optimization_normal = normalize(Point3D(0.0, 1.0, -1.0))
    #optimization_normal = normalize(Point3D(1.0, 0.0, 0.0))
    
    ##testing out new growth function:
    #scale = new_explore_direction(screen_normal, center_scale, principal_ray, light_radius, normalize(Point3D(0.0, 1.0, -1.0)))
    #ordered_scales = [center_scale, scale]
    #for scale in ordered_scales:
    #    scale.ensure_mesh()
    #    
    ##a bit of a hack so we can visualize the real error:
    #center_scale.adjacent_scale = scale
    #center_scale.screen_normal = screen_normal
    
    ##testing for optimization:
    #optics.utils.profile_line('new_explore_direction(screen_normal, center_scale, principal_ray, light_radius, normalize(Point3D(0.0, 1.0, -1.0)))', globals(), locals())
    #import sys
    #sys.exit()
    
    def grow_in_direction(direction, new_scales, optimization_normal):
        phi = 0.0
        prev_scale = center_scale
        while phi < final_phi:
            #phi += phi_step
            #theta = math.pi / 2.0
            #if direction < 0:
            #    theta = 3.0 * math.pi / 2.0
            #angle_vec = AngleVector(theta, phi)
            
            #old function: used to optimize in multiple directions
            #TODO: put something like it back to have a curved screen
            #scale, error = optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)            
            #scale, error = explore_direction(direction * optimization_normal, lower_bound, upper_bound, prev_scale, principal_ray, light_radius, angle_vec)
            #scale, error = optics.parallel.call_via_pool(process_pool, explore_direction, [direction * optimization_normal, lower_bound, upper_bound, prev_scale, principal_ray, light_radius, angle_vec])
            #scale, error = explore_direction(direction * optimization_normal, lower_bound, upper_bound, prev_scale, principal_ray, light_radius, angle_vec)
            
            start_time = time.time()
            
            nearby_scales = [prev_scale]
            scale = optics.parallel.call_via_pool(process_pool, new_explore_direction, [screen_normal, prev_scale, nearby_scales, principal_ray, light_radius, direction*optimization_normal])
            
            if stop_flag.is_set():
                return new_scales
            
            #evaluate the new scale
            pixel_errors = evaluate_scale(scale, prev_scale, light_radius)
            
            new_scales.append(scale)
            on_new_scale(scale)
            prev_scale = scale
            phi = scale.angle_vec.phi
            print("Finished (phi=%.4f,theta=%.4f) in %.3f    [errors=%s]" % (phi, scale.angle_vec.theta, time.time() - start_time, pixel_errors))
        return new_scales

    upward_arc = []
    rightward_arc = []
    threads = [threading.Thread(target=grow_in_direction, args=args) for args in (\
        (1.0, upward_arc, normalize(Point3D(0.0, 1.0, -1.0))),
        (1.0, rightward_arc, normalize(Point3D(1.0, 0.0, 0.0)))
    )]
    
    #leftward_arc = []
    #rightward_arc = []
    #threads = [threading.Thread(target=grow_in_direction, args=args) for args in (\
    #    (1.0, leftward_arc, normalize(Point3D(-1.0, 0.0, 0.0))),
    #    (1.0, rightward_arc, normalize(Point3D(1.0, 0.0, 0.0)))
    #)]
    
    for thread in threads:
        thread.start()
        
    while True:
        all_threads_done = True not in [t.is_alive() for t in threads]
        if all_threads_done:
            break
    
        if stop_flag.is_set():
            return []
        
        time.sleep(0.1)
        
    for thread in threads:
        thread.join()
        
    #leftward_arc.reverse()
    #ordered_scales = leftward_arc + [center_scale] + rightward_arc
    
    def grow_quadrant(vertical_arc, horizontal_arc, scale_rows, optimization_normal):
        """
        Builds it up, horizontally row by row, from the center.
        """
        diagonal_scale = center_scale
        for start_scale in vertical_arc:
            new_horizontal_arc = []
            left_scale = start_scale
            for bottom_scale in horizontal_arc:
                prev_scale = left_scale
                nearby_scales = [diagonal_scale, left_scale, bottom_scale]
                start_time = time.time()
                #TODO: obviously put this back
                #new_scale = optics.parallel.call_via_pool(process_pool, new_explore_direction, [screen_normal, prev_scale, nearby_scales, principal_ray, light_radius, optimization_normal])
                new_scale = new_explore_direction(screen_normal, prev_scale, nearby_scales, principal_ray, light_radius, optimization_normal)
                
                if stop_flag.is_set():
                    return scale_rows
                
                pixel_errors = evaluate_scale(new_scale, prev_scale, light_radius)
                new_horizontal_arc.append(new_scale)
                on_new_scale(new_scale)
                print("Finished (phi=%.4f,theta=%.4f) in %.3f    [errors=%s]" % (new_scale.angle_vec.phi, new_scale.angle_vec.theta, time.time() - start_time, pixel_errors))
                
                left_scale = new_scale
                diagonal_scale = bottom_scale
            diagonal_scale = start_scale
            horizontal_arc = new_horizontal_arc
            scale_rows.append(new_horizontal_arc)
        return scale_rows
    
    final_scale_rows = []
    grow_quadrant(upward_arc, rightward_arc, final_scale_rows, normalize(Point3D(1.0, 0.0, 0.0)))
    ordered_scales = upward_arc + rightward_arc + [center_scale]
    all_scale_rows = [[center_scale] + rightward_arc]
    for i in range(0, len(final_scale_rows)):
        row = final_scale_rows[i]
        ordered_scales += row
        all_scale_rows.append([upward_arc[i]] + row)
        
    arcs = create_patch_arcs(all_scale_rows, light_radius, step_size=0.5)
    #to get the surface oriented the correct direction
    arcs.reverse()
    mesh = optics.mesh.Mesh(mesh=optics.mesh.mesh_from_arcs(arcs))
    mesh.export("new_shell.stl")
    
    ##print out a little graph of the errors of the scales so we can get a sense
    #downward_arc.reverse()
    #ordered_scales = downward_arc + [center_scale] + upward_arc
    #print("theta  phi     focal_error")
    #for scale in ordered_scales:
    #    scale.ensure_mesh()
    #    #scale._calculate_rays()
    #    print("%.2f   %.2f      %.5f" % (scale.angle_vec.theta, scale.angle_vec.phi, scale.focal_error))
    #    
    ###a bit of a hack so we can visualize the real error:
    ##center_scale.adjacent_scale = upward_arc[0]
    ##center_scale.screen_normal = screen_normal
    #    
    ##export all of the scales as one massive STL
    #meshes = [x._mesh for x in ordered_scales]
    #merged_mesh = optics.mesh.merge_meshes(meshes)
    #optics.mesh.Mesh(mesh=merged_mesh).export("all_scales.stl")
    #
    ##export the shape formed by the screen pixels as an STL
    #create_screen_mesh(ordered_scales).export("screen.stl")
    
    return ordered_scales

#TODO: will have to put all of these arcs together in an outer loop and then pass to the mesher
#note: should include original arc scales in these rows, and the center one
def create_patch_arcs(scale_rows, light_radius, step_size=0.5):
    """
    create a set of arcs through these scales, for use when meshing
    """
    
    #walk through the first vertical column and generate starting points for each arc
    arcs = []
    for i in range(0, len(scale_rows)-1):
        scale = scale_rows[i][0]
        next_scale = scale_rows[i+1][0]
        up_vector = normalize(next_scale.shell_point - scale.shell_point)
        end_point, arc = scale._get_arc_and_start(scale.shell_point, up_vector, step_size)
        #remove the last point if it is too close to the next shell point
        if numpy.linalg.norm(arc[-1] - next_scale.shell_point) < step_size:
            arc.pop(0)
        arcs.append(arc)
    #just stick the last shell point on there for completeness
    arcs[-1].append(next_scale.shell_point)
    #concatenate all of the arcs
    vertical_arc = numpy.concatenate(arcs)
    
    #create all of the horizontal arcs
    segments_per_shell = light_radius / step_size #note: this is not EXACTLY going to be the step size, but will be close
    arcs = []
    current_row_idx = 0
    for start_point in vertical_arc:
        arc = []
        #if we're closer to the next row, use that one instead
        if current_row_idx < len(scale_rows) - 1:
            if dist2(start_point, scale_rows[current_row_idx+1][0].shell_point) < dist2(start_point, scale_rows[current_row_idx][0].shell_point):
                current_row_idx += 1
        current_row = scale_rows[current_row_idx]
        #walk across the row and extend the arc by a fixed number of segments per shell
        for i in range(1, len(current_row)):
            scale = current_row[i-1]
            next_scale = current_row[i]
            ray_start = scale.shell_point
            ray_end = next_scale.shell_point
            ray_vector = ray_end - ray_start
            ray_length = numpy.linalg.norm(ray_vector)
            ray_normal = ray_vector / ray_length
            distances = numpy.linspace(0, ray_length, num=segments_per_shell, endpoint=False)
            
            #transform the ray into scale coordinates
            transformed_arc_start = scale._world_to_local(start_point)
            transformed_ray_normal = scale._world_to_local_rotation.dot(ray_normal)
            
            #evaluate each point
            for dist in distances:
                point = dist * transformed_ray_normal + transformed_arc_start
                point = Point3D(point[0], point[1], scale._poly.eval_poly(point[0], point[1]))
                arc.append(scale._local_to_world(point))
            start_point = arc[-1]
        arcs.append(arc)
    
    return arcs
    
    
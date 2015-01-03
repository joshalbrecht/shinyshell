#!/usr/bin/python

"""
Installation instructions: sudo pip install pyglet
(actually make sure you install everything from main.py as well)

This application is meant to be a complete replacement for main.py (except I'm trying to never use pyoptools)
All assumptions and coordinates are the same as in main.py.

Usage instructions:

left click and drag to move screen pixels or shell sections
right click to make a new piece of shell (and paired pixel)
middle mouse drag to pan
middle mouse roll to zoom
shift + middle mouse to rotate the world
"""

import math
import sys
import itertools
import time

import OpenGL.GL
import OpenGL.GLU
import pyglet
import pyglet.window.key
import pyglet.window.mouse
import numpy
import scipy.integrate
import scipy.optimize
import pyximport; pyximport.install()
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

#this is the one thing that is allowed to import *
from optics.base import *
import optics.globals
import optics.rotation_matrix
import optics.mesh
import optics.calculations
import hacks.taylor_poly
import viewer.window

class Scale(optics.mesh.Mesh):
    def __init__(self, shell_point=None, pixel_point=None, angle_vec=None, **kwargs):
        optics.mesh.Mesh.__init__(self, **kwargs)
        assert shell_point != None
        assert pixel_point != None
        assert angle_vec != None
        self._shell_point = shell_point
        self._pixel_point = pixel_point
        self._angle_vec = angle_vec
        self.shell_distance_error = None
        self._num_rays = 11
        self._pupil_radius = 3.0
        
        self._rays = None
        
    @property
    def shell_point(self):
        return self._shell_point
    
    @property
    def pixel_point(self):
        return self._pixel_point
    
    @property
    def angle_vec(self):
        return self._angle_vec
    
    def points(self):
        point_data = self._mesh.GetPoints().GetData()
        return [numpy.array(point_data.GetTuple(i)) for i in range(0, point_data.GetSize())]
    
    def render(self):
        optics.mesh.Mesh.render(self)
        
        if self._rays == None:
            self._calculate_rays()
        
        for ray in self._rays:
            ray.render()
            
    def _calculate_rays(self):
        infinite_rays = []
        base_eye_ray = Ray(Point3D(0,0,0), 1.2 * self._shell_point)
        if self._num_rays == 1:
            infinite_rays.append(base_eye_ray)
        else:
            for y in numpy.linspace(-self._pupil_radius, self._pupil_radius, num=self._num_rays):
                delta = Point3D(0, y, 0)
                infinite_rays.append(Ray(base_eye_ray.start + delta, base_eye_ray.end+delta))
        
        #TEMP: just want to see how close we're getting to the correct pixel location:
        screen_plane = Plane(self._pixel_point, normalize(self.shell_point - self.pixel_point))
        
        reflection_length = 1.1 * numpy.linalg.norm(self.shell_point - self.pixel_point)
        self._rays = []
        for ray in infinite_rays:
            intersection, normal = self.intersection_plus_normal(ray.end, ray.start)
            normal *= -1.0
            if intersection != None:
                self._rays.append(LightRay(ray.start, intersection))
                reverse_ray_direction = normalize(ray.start - ray.end)
                midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
                reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
                ray_to_screen = LightRay(intersection, intersection + reflection_length * reflection_direction)
                self._rays.append(ray_to_screen)
                
                plane_intersection = screen_plane.intersect_line(ray_to_screen.start, ray_to_screen.end)
                print numpy.linalg.norm(plane_intersection - self._pixel_point)
                
class PolyScale(Scale):
    def __init__(self,  poly=None,
                        world_to_local_rotation=None,
                        local_to_world_rotation=None,
                        world_to_local_translation=None,
                        domain_cylinder_point=None,
                        domain_cylinder_radius=None,
                        **kwargs):
        Scale.__init__(self, **kwargs)
        self._poly = poly
        self._world_to_local_rotation = world_to_local_rotation
        self._local_to_world_rotation = local_to_world_rotation
        self._world_to_local_translation = world_to_local_translation
        self._local_to_world_translation = -1.0 * world_to_local_translation
        self._domain_cylinder_point = domain_cylinder_point
        self._domain_cylinder_radius = domain_cylinder_radius
        
    def render(self):
        if self._mesh == None:
            self.set_mesh(self._create_mesh())
        Scale.render(self)
        
    def _local_to_world(self, p):
        return self._local_to_world_rotation.dot(p) + self._local_to_world_translation
        
    def _world_to_local(self, p):
        return self._world_to_local_rotation.dot(p + self._world_to_local_translation)
    
    def _create_mesh(self):
        """
        Just makes an approximate mesh. For rendering mostly.
        """
        #TODO: probably more robust to find the min and max in each direction (for x and y) before we are out of domain
        #for now we just assume that you're probably not going to go beyond 2.0 * light radius in either direction
        #make arcs along each of the possible x values
        multiplier = 2.0
        step = 0.5
        arcs = []
        for x in numpy.arange(-multiplier * self._domain_cylinder_radius, multiplier * self._domain_cylinder_radius, step):
            arc = []
            for y in numpy.arange(-multiplier * self._domain_cylinder_radius, multiplier * self._domain_cylinder_radius, step):
                z = self._poly.eval_poly(x, y)
                point = self._local_to_world(Point3D(x, y, z))
                arc.append(point)
            arcs.append(arc)
        
        #TODO: rename the mesh module. it's too generic of a name
        #make a mesh from those arcs
        base_mesh = optics.mesh.mesh_from_arcs(arcs)
        
        #trim the mesh given our domain cylinder
        trimmed_mesh = optics.mesh.trim_mesh_with_cone(base_mesh, Point3D(0.0, 0.0, 0.0), normalize(self._shell_point), self._domain_cylinder_radius)
        
        return trimmed_mesh

    #TODO: filter out things that do not collide within our domain
    #TODO: make another function that only returns the intersection (for efficiency, since this is in the critical path)
    def intersection_plus_normal(self, start, end):
        """
        Really the entire reason we switch to taylor poly instead. much better intersections hopefully...
        """
        #translate start and end into local coordinates
        transformed_start = self._world_to_local(start)
        transformed_end = self._world_to_local(end)
        #use the cython collision function to figure out where we collided
        #TODO: unsure if we actually need to normalize...
        point = self._poly._intersection(transformed_start, normalize(transformed_end-transformed_start))
        #calculate the normal as well
        normal = -1.0 * self._poly.normal(point)
        return self._local_to_world(point), self._local_to_world_rotation.dot(normal)
    
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
def make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec):
    """
    returns a non-trimmed scale patch based on the point (where the shell should be centered)
    angle_vec is passed in for our convenience, even though it is duplicate information (given the shell_point)
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
    #local_to_world_rotation = numpy.linalg.inv(world_to_local_rotation)
    local_to_world_rotation = numpy.zeros((3, 3))
    optics.rotation_matrix.R_2vect(local_to_world_rotation, Point3D(0.0, 0.0, 1.0), z_axis_world_dir)
    
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
    coefficients = polyfit2d(x, y, z, order=10)
    order = int(numpy.sqrt(len(coefficients)))
    cohef = []
    for i in range(0, order):
        cohef.append(coefficients[i*order:(i+1)*order])
    cohef = numpy.array(cohef).copy(order='C')
    poly = hacks.taylor_poly.TaylorPoly(cohef=cohef.T)
    
    scale = PolyScale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        poly=poly,
        world_to_local_rotation=world_to_local_rotation,
        local_to_world_rotation=local_to_world_rotation,
        world_to_local_translation=world_to_local_translation,
        domain_cylinder_radius=light_radius
    )
    return scale

def make_old_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec):

    spine = optics.calculations.create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=False)
    ribs = []
    for point in spine:
        rib = optics.calculations.create_arc(principal_ray, point, screen_point, light_radius, angle_vec, is_horizontal=True)
        ribs.append(numpy.array(rib))
        
    #TODO: replace this with original:
    return Scale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        mesh=optics.mesh.mesh_from_arcs(ribs)
    )

    #scale = mesh.Mesh(mesh.mesh_from_arcs(ribs))
    ##scale.export("temp.stl")
    #trimmed_scale = Scale(
    #    shell_point=shell_point,
    #    pixel_point=screen_point,
    #    angle_vec=angle_vec,
    #    mesh=optics.mesh.trim_mesh_with_cone(scale._mesh, Point3D(0.0, 0.0, 0.0), normalize(shell_point), light_radius)
    #)
    ##trimmed_scale.export("temp.stl")
    #return trimmed_scale
    
def calculate_error(scale, reference_scale):
    """
    I guess shoot rays all over the scale (from the pixel location), and see which also hit the reference scale, and get the distance
    least squares? or just sum all of it? I wonder why people use least squares all the time...
    note: will have to be average error per sample point, since different shells will have different number of sample points
    question is just whether to average the squares, or regularize them
    """
    start = reference_scale.pixel_point
    dist = 0.0
    num_hits = 0
    points = reference_scale.points()
    for point in points:
        end = 2.0 * (point - start) + start
        intersection_point, intersection_normal = scale.intersection_plus_normal(start, end)
        if intersection_point != None:
            num_hits += 1
            #print numpy.linalg.norm(intersection_point - point)
            dist += numpy.linalg.norm(intersection_point - point)
            #delta = intersection_point - point
            #dist += delta.dot(delta)
    average_error = dist / num_hits
    #print num_hits
    return average_error

def _get_scale_and_error_at_distance(distance, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec):
    shell_point = distance * angle_normal
    scale = make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec)
    error = max([calculate_error(scale, reference_scale) for reference_scale in reference_scales])
    scale.shell_distance_error = error
    return scale, error

#TODO: there are almost certainly better ways to optimize something like this, see numpy
def find_scale_and_error_at_best_distance(reference_scales, principal_ray, screen_point, light_radius, angle_vec):
    """
    iteratively find the best distance that this scale can be away from the reference scales
    """
    #seems pretty arbitrary, but honestly at that point the gains here are pretty marginal
    num_iterations = 14
    if optics.globals.LOW_QUALITY_MODE:
        num_iterations = 8
    angle_normal = angle_vector_to_vector(angle_vec, principal_ray)
    reference_distance = numpy.linalg.norm(reference_scales[0].shell_point)
    
    lower_bound_dist = reference_distance - light_radius
    upper_bound_dist = reference_distance + light_radius
    
    scales = {}
    def f(x):
        scale, error = _get_scale_and_error_at_distance(x, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec)
        scales[x] = (scale, error)
        return error
    #best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=0.001, full_output=True, disp=0)
    best_value = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=0.001, full_output=False, disp=0)
    return scales[best_value]
    
def explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec):
    results = {}
    def f(x):
        pixel_point = prev_scale.pixel_point + x * optimization_normal
        scale, error = find_scale_and_error_at_best_distance([prev_scale], principal_ray, pixel_point, light_radius, angle_vec)
        results[x] = (scale, error)
        return error
    best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound, upper_bound, maxfun=num_iterations, xtol=0.0001, full_output=True, disp=3)
    return results[best_value]

def optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec):
    approximately_correct_scale, decent_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #print("Decent error: " + str(decent_error))
    
    #after that, simply find the point along that line (from the shell to that pixel) that is closest to the previous pixel
    #(since we don't want the screen to get any bigger than it has to)
    #and make the shell there
    #TODO: will have to look at how the optimization curves look for surfaces where we are optimizing against 3 surfaces...
    #might have to do another call to "explore_direction" to get the absolute best performance
    best_screen_point = closestPointOnLine(prev_scale.pixel_point, approximately_correct_scale.pixel_point, approximately_correct_scale.shell_point)
    #best_scale, error_for_best_scale = find_scale_and_error_at_best_distance([prev_scale], principal_ray, best_screen_point, light_radius, angle_vec)
    #print("'best' error: " + str(error_for_best_scale))
    
    #doing another crawl along the line because why not
    optimization_normal = normalize(best_screen_point - prev_scale.pixel_point)
    final_scale, final_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #print("Final error: " + str(final_error))
    #scales.append(final_scale)
    return final_scale, final_error
    
def create_screen_mesh(ordered_scales):
    arc = [scale.pixel_point for scale in ordered_scales]
    left_arc = [p + Point3D(-1.0, 0.0, 0.0) for p in arc]
    right_arc = [p + Point3D(1.0, 0.0, 0.0) for p in arc]
    return mesh.Mesh(mesh=mesh.mesh_from_arcs([right_arc, arc, left_arc]))
    
def create_surface_via_scales(initial_shell_point, initial_screen_point, principal_ray):
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
    fov = math.pi / 2.0
    #per whole the screen. So 90 steps for a 90 degree total FOV would be one step per degree
    total_phi_steps = 90
    
    #calculated:
    total_vertical_resolution = 2000
    min_pixel_spot_size = 0.005
    min_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * min_pixel_spot_size
    max_pixel_spot_size = 0.015
    max_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * max_pixel_spot_size
    
    #TODO: go delete the old scale stuff if this works!!!!!
    
    #create the first scale
    center_scale = make_old_scale(principal_ray, initial_shell_point, initial_screen_point, light_radius, AngleVector(0.0, 0.0))
    center_scale.shell_distance_error = 0.0
    #scales = [center_scale]
    scales = []
    
    new_center_scale = make_scale(principal_ray, initial_shell_point, initial_screen_point, light_radius, AngleVector(0.0, 0.0))
    new_center_scale.shell_distance_error = 0.0
    scales.append(new_center_scale)
    
    #create another scale right above it for debugging the error function
    #shell_point = initial_shell_point + Point3D(0.0, 3.0, -1.0)
    #angle_vec = AngleVector(math.pi/2.0, normalized_vector_angle(principal_ray, normalize(shell_point)))
    #other_scale = make_scale(principal_ray, shell_point, initial_screen_point+Point3D(0.0, -min_pixel_spot_size, min_pixel_spot_size), light_radius, angle_vec)
    #scales = [center_scale, other_scale]
    
    #calculate_error(other_scale, center_scale)
    
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
    
    ##ok, new approach to actually optimizing the next shell:
    ##simply walk along the direction orthogonal to the last pixel -> shell vector in the current plane
    ##and find the location with the minimal error
    #
    #lower_bound = 0.0
    ##NOTE: is a hack / guestimate
    #upper_bound = 2.0 * light_radius
    #num_iterations = 16
    #if optics.globals.LOW_QUALITY_MODE:
    #    num_iterations = 8
    #    
    #phi_step = 0.05
    #final_phi = fov/2.0
    #
    #for direction in (1.0, -1.0):
    #    phi = 0.0
    #    prev_scale = center_scale
    #    while phi < final_phi:
    #        phi += phi_step
    #        theta = math.pi / 2.0
    #        if direction < 0:
    #            theta = 3.0 * math.pi / 2.0
    #        angle_vec = AngleVector(theta, phi)
    #        #TODO: obviously this has to change in the general case
    #        optimization_normal = direction * numpy.cross(Point3D(1.0, 0.0, 0.0), normalize(prev_scale.shell_point - prev_scale.pixel_point))
    #        scale, error = optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #        scales.append(scale)
    #        prev_scale = scale
    #        
    ##print out a little graph of the errors of the scales so we can get a sense
    ##NOTE: this shuffling is just so that the errors are printed in an intuitive order
    #num_scales = len(scales)
    #num_scales_in_arc = (num_scales - 1) / 2
    #lower_arc = scales[num_scales_in_arc+1:]
    #lower_arc.reverse()
    #ordered_scales = lower_arc + scales[:num_scales_in_arc+1]
    #print("theta  phi     error")
    #for scale in ordered_scales:
    #    print("%.2f %.2f    %.5f" % (scale.angle_vec.theta, scale.angle_vec.phi, scale.shell_distance_error))
    #    
    ##export all of the scales as one massive STL
    #merged_mesh = mesh.merge_meshes(ordered_scales)
    #mesh.Mesh(mesh=merged_mesh).export("all_scales.stl")
    #
    ##export the shape formed by the screen pixels as an STL
    #create_screen_mesh(ordered_scales).export("screen.stl")
    
    return scales

def plot_error(x, y, z):
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig)
    p = ax.plot_wireframe(x, y, z)
    plt.show()

def main():
    win = viewer.window.Window(23) # set the fps
    initial_shell_point = Point3D(0.0, 0.0, -60.0)
    initial_screen_point = Point3D(0.0, 40.0, -20.0)
    principal_ray = Point3D(0.0, 0.0, -1.0)
    win.scales = create_surface_via_scales(initial_shell_point, initial_screen_point, principal_ray)
    win.run()

if __name__ == '__main__':
    main()
    
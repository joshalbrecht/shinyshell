
"""
An attempt to rewrite using patches, and not mixed in with a bunch of dead code
Goal is to be able to delete calculations.py and new_calculations.py when finished with this one
"""

import math

import scipy.optimize

from optics.base import * # pylint: disable=W0401,W0614
import optics.globals
import optics.utils
import optics.parallel
import optics.scale
import optics.arc
import optics.arcplane
import optics.localpoly
import optics.patch
import optics.rotation_matrix

LIGHT_RADIUS = 3.0
FORCE_FLAT_SCREEN = False
FOV = 70.0 * math.pi / 180.0

def grow_surface(initial_shell_point, initial_screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_patch):
    """
    Has the pool of workers continually create new patches. Each patch completed enables other, new patches to be created.
    
    For now this is just a single thread, but it's meant to be parallelize nicely
    """
    
    max_angle = FOV/2.0
    angle_step = math.fabs(math.atan(optics.globals.LIGHT_RADIUS / initial_shell_point[2]))
    
    #a simple queue of requests in the form of arguments to pass to create_patch
    patch_requests = []
    
    all_patches = []
    patch_grid = {}
    def grid_index(patch):
        """
        Returns a (mu, rho) integer pair representing the patch. For easy indexing and tracking which have been created.
        """
        mu = round(math.fabs(patch.mu_start_plane.angle) / angle_step)
        if patch.mu_end_plane.angle < 0:
            if patch.mu_start_plane.angle == 0.0:
                mu = -1
            else:
                mu += 2
                mu *= -1
        rho = round(math.fabs(patch.rho_start_plane.angle) / angle_step)
        if patch.rho_end_plane.angle < 0:
            if patch.rho_start_plane.angle == 0.0:
                rho = -1
            else:
                rho += 2
                rho *= -1
        return (mu, rho)
    
    def _create_patch_for_index(index):
        """
        Create a patch at a given index. Assumes that any required previous patches exist.
        """
        mu_idx, rho_idx = index
        prev_rho_patch = None
        prev_mu_patch = None
        if mu_idx == 0 and rho_idx == 0:
            shell_point = initial_shell_point
            screen_point = initial_screen_point
            mu_start_plane = optics.arcplane.ArcPlane(mu=0.0)
            mu_end_plane = optics.arcplane.ArcPlane(mu=angle_step)
            rho_start_plane = optics.arcplane.ArcPlane(rho=0.0)
            rho_end_plane = optics.arcplane.ArcPlane(rho=angle_step)
        else:
            if mu_idx != 0:
                if mu_idx > 0:
                    prev_mu_idx = mu_idx - 1
                else:
                    prev_mu_idx = mu_idx + 1
                prev_mu_patch = patch_grid[(prev_mu_idx, rho_idx)]
                mu_start_plane = prev_mu_patch.mu_end_plane
                if mu_idx > 0:
                    mu_end_plane = optics.arcplane.ArcPlane(mu=angle_step * (mu_idx+1))
                else:
                    mu_end_plane = optics.arcplane.ArcPlane(mu=angle_step * mu_idx)
            else:
                mu_start_plane = optics.arcplane.ArcPlane(mu=0.0)
                mu_end_plane = optics.arcplane.ArcPlane(mu=angle_step)
            if rho_idx != 0:
                if rho_idx > 0:
                    prev_rho_idx = rho_idx - 1
                else:
                    prev_rho_idx = rho_idx + 1
                prev_rho_patch = patch_grid[(mu_idx, prev_rho_idx)]
                rho_start_plane = prev_rho_patch.rho_end_plane
                if rho_idx > 0:
                    rho_end_plane = optics.arcplane.ArcPlane(rho=angle_step * (rho_idx+1))
                else:
                    rho_end_plane = optics.arcplane.ArcPlane(rho=angle_step * rho_idx)
            else:
                rho_start_plane = optics.arcplane.ArcPlane(rho=0.0)
                rho_end_plane = optics.arcplane.ArcPlane(rho=angle_step)
            prev_patches = [patch for patch in (prev_mu_patch, prev_rho_patch) if patch != None]
            
            #figure out the screen and shell point
            shell_point = get_shell_point(prev_patches, mu_start_plane.angle, rho_start_plane.angle)
            screen_point = get_focal_point(prev_patches, shell_point)
            
        return optics.patch.create_patch(
            shell_point,
            screen_point,
            mu_start_plane,
            mu_end_plane,
            rho_start_plane,
            rho_end_plane,
            prev_mu_patch,
            prev_rho_patch
        )
    
    def on_patch_completed(patch):
        """
        Figures out which new patch requests to create and add to the list of requests
        """
        on_new_patch(patch)
        all_patches.append(patch)
        
        #stick them into a grid (because we need to know when adjacent ones have been enabled)
        (mu, rho) = grid_index(patch)
        patch_grid[(mu, rho)] = patch
        for offset_mu, offset_rho in (
                ( 1,  1),
                ( 1, -1),
                (-1, -1),
                (-1,  1),
            ):
            if (mu+offset_mu, rho+offset_rho) not in patch_grid:
                if (mu+offset_mu, rho) in patch_grid and (mu, rho+offset_rho) in patch_grid:
                    patch_requests.append((mu+offset_mu, rho+offset_rho))
        
        #enable horizontal and vertical arcs
        if patch.mu_start_plane.angle == 0.0 and math.fabs(patch.mu_start_plane.angle) < max_angle:
            if (mu, rho-1) not in patch_grid:
                patch_requests.append((mu, rho-1))
            if (mu, rho+1) not in patch_grid:
                patch_requests.append((mu, rho+1))
        if patch.rho_start_plane.angle == 0.0 and math.fabs(patch.rho_start_plane.angle) < max_angle:
            if (mu-1, rho) not in patch_grid:
                patch_requests.append((mu-1, rho))
            if (mu+1, rho) not in patch_grid:
                patch_requests.append((mu+1, rho))

    #start off the process with the first patch
    central_patch = _create_patch_for_index((0.0, 0.0))
    on_patch_completed(central_patch)
    
    #continuously process new patches while there are any remaining requests
    while len(patch_requests) > 0:
        request = patch_requests.pop(0)
        on_patch_completed(_create_patch_for_index(request))
        
    return all_patches
    
def get_shell_point(patches, mu, rho):
    """
    Figures out where the next shell point is (at this given mu and rho) based
    on previous patches.
    """
    point = Point3D(0.0, 0.0, 0.0)
    for patch in patches:
        point += patch.get_corner(mu, rho)
    return point / len(patches)

def get_focal_point(patches, shell_point, num_rays=5):
    """
    Figures out where the patches focus light that is directed at this shell point
    """
    focal_point = Point3D(0.0, 0.0, 0.0)
    for patch in patches:
        #create a bunch of parallel rays coming from the eye
        ray_vector = normalize(shell_point)
        ray_rotation = numpy.zeros((3, 3))
        optics.rotation_matrix.R_2vect(ray_rotation, PRINCIPAL_RAY, ray_vector)
        rays = []
        for x in numpy.linspace(-LIGHT_RADIUS, LIGHT_RADIUS, num_rays*2+1):
            for y in numpy.linspace(-LIGHT_RADIUS, LIGHT_RADIUS, num_rays*2+1):
                start_point = ray_rotation.dot(Point3D(x, y, 0.0))
                rays.append(Ray(start_point, start_point + ray_vector))
                
        #find the point such that the spot size is minimized on the screen.
        #can average the normal of the reflected rays to get approximately where the screen goes
        #then iteratively try different distances until we've minimized the spot size there
        focal_point = Point3D(0.0, 0.0, 0.0)
        reflected_rays = patch.reflect_rays_no_bounds(rays)
        approximate_screen_normal = sum([normalize(ray.start - ray.end) for ray in reflected_rays]) / len(reflected_rays)
        def calculate_spot_size(distance):
            """
            :returns: average distance from the central point for the plane at this distance
            """
            screen_plane = Plane(distance * approximate_screen_normal * -1.0 + shell_point, approximate_screen_normal)
            points = []
            for ray in reflected_rays:
                points.append(screen_plane.intersect_line(ray.start, ray.end))
            average_point = sum(points) / len(points)
            errors = [numpy.linalg.norm(p - average_point) for p in points]
            return sum(errors) / len(errors)
        previous_distance = numpy.linalg.norm(patch.shell_point - patch.screen_point)
        min_dist = previous_distance * 0.9
        max_dist = previous_distance * 1.1
        num_iterations = 20
        tolerance = 0.0001
        best_dist = scipy.optimize.fminbound(calculate_spot_size, min_dist, max_dist, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
        focal_point += best_dist * approximate_screen_normal * -1.0 + shell_point
    return focal_point / len(patches)
    
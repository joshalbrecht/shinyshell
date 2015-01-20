
"""
An attempt to rewrite using patches, and not mixed in with a bunch of dead code
Goal is to be able to delete calculations.py and new_calculations.py when finished with this one
"""

import math

import scipy.integrate
import scipy.optimize

from optics.base import * # pylint: disable=W0401,W0614
import optics.globals
import optics.utils
import optics.parallel
import optics.scale
import optics.arc
import optics.arcplane

FORCE_FLAT_SCREEN = False
FOV = 70.0 * math.pi / 180.0

ORIGIN = Point3D(0.0, 0.0, 0.0)
FORWARD = 1.0
BACKWARD = -1.0

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
            
        return create_patch(
            shell_point,
            screen_point,
            rho_start_plane,
            rho_end_plane,
            mu_start_plane,
            mu_end_plane,
            prev_rho_patch,
            prev_mu_patch
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

def create_patch(
        shell_point,
        screen_point,
        rho_start_plane,
        rho_end_plane,
        mu_start_plane,
        mu_end_plane,
        prev_rho_patch=None,
        prev_mu_patch=None
        ):
    """
    Given a bunch of initial parameters, create a new part of the surface.
    """
    #figure out which direction we are growing (in the rho and mu directions)
    rho_direction = blah
    mu_direction = blah
    
    #either use the previous (neighboring) arc, or create it if there is no patch there because this is the zero edge
    #do in both directions
    if prev_mu_patch != None:
        prev_mu_arc = prev_mu_patch.get_edge_arc(mu_direction)
    else:
        prev_mu_arc = grow_arc(shell_point, screen_point, mu_start_plane)
    if prev_rho_patch != None:
        prev_rho_arc = prev_rho_patch.get_edge_arc(rho_direction)
    else:
        prev_rho_arc = grow_arc(shell_point, screen_point, rho_start_plane)
    
    #create a patch. not fully initialized
    patch = optics.patch.Patch(shell_point, screen_point, pre_rho_arc, prev_mu_arc, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane)
    
    #create a grid of points for two conflicting sets of ribs
    vertical_grid = patch.make_grid(True)
    horizontal_grid = patch.make_grid(False)
    
    #optimize performance of the surface by finding the right mix of points to minimize focal error in both directions
    grids = {}
    def f(weight):
        """
        Given a weighting between vertical and horizontal ribs, returns the error for the surface.
        Error is calculated as max of x and y error.
        Error is about how well rays along the extreme edges are focused to the screen point.
        """
        #make weighted average of each point in the grid
        grid = blah
        grids[weight] = grid
        
        #include the points from existing arcs and shell point (weighted?)
        
        #fit taylor poly to the points
        #TODO: need to make sure this is extremely accurate
        #if it isn't, then we have to ensure that we're using the taylor poly and casting rays to create the other edge arcs instead
        #because the point grid won't have enough accuracy
        
        #cast rays against the taylor poly to measure x and y error
        x_error = blah
        y_error = blah
        error = max((x_error, y_error))
        print error
        return error
    num_iterations = 20
    tolerance = 0.0001
    best_weight = scipy.optimize.fminbound(f, 0.0, 1.0, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    patch.grid = grids[best_weight]
    
    return patch
    
def grow_arc(shell_point, screen_point, arc_plane):
    blah
    
def get_shell_point(patches, mu, rho):
    """
    Figures out where the next shell point is (at this given mu and rho) based
    on previous patches.
    """
    point = Point3D(0.0, 0.0, 0.0)
    for patch in patches:
        point += patch.get_corner(mu, rho)
    return point / len(patches)

def get_focal_point(patches, shell_point):
    """
    Figures out where the patches focus light that is directed at this shell point
    """
    focal_point = Point3D(0.0, 0.0, 0.0)
    for patch in patches:
        rays = blah
        reflected_rays = patch.reflect_rays_no_bounds(rays)
        point = find_closest_point(reflected_rays)
        focal_point += point
    return focal_point / len(patches)
    

    
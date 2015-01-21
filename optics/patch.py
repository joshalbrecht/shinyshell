
import scipy.integrate
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
    This is an initialization function for Patch
    """
    #figure out which direction we are growing (in the rho and mu directions)
    rho_direction = FORWARD
    if rho_start_plane.angle < rho_end_plane.angle:
        rho_direction = BACKWARD
    mu_direction = FORWARD
    if mu_start_plane.angle < mu_end_plane.angle:
        mu_direction = BACKWARD
    
    #either use the previous (neighboring) arc, or create it if there is no patch there because this is the zero edge
    #do in both directions
    if prev_mu_patch != None:
        prev_mu_arc = prev_mu_patch.get_edge_arc(False, mu_direction)
    else:
        prev_mu_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0)
    if prev_rho_patch != None:
        prev_rho_arc = prev_rho_patch.get_edge_arc(True, rho_direction)
    else:
        prev_rho_arc = optics.arc.new_grow_arc(shell_point, screen_point, mu_start_plane, rho_end_plane, previous_normal_function=None, falloff=-1.0)
    
    #create a patch. not fully initialized
    patch = optics.patch.Patch(shell_point, screen_point, prev_rho_arc, prev_mu_arc, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane)
    
    #create a grid of points for two conflicting sets of ribs
    vertical_grid = patch.make_grid(True)
    horizontal_grid = patch.make_grid(False)
    
    #convert the grids to local space for this patch so we can fit taylor polys
    shell_point_to_eye_normal = normalize(ORIGIN - shell_point)
    shell_point_to_screen_normal = normalize(screen_point - shell_point)
    shell_point_surface_normal = normalize(shell_point_to_eye_normal + shell_point_to_screen_normal)
    space = CoordinateSpace(shell_point, shell_point_surface_normal)
    project = numpy.vectorize(space.point_to_space)
    projected_vertical_grid = project(vertical_grid)
    projected_horizontal_grid = project(horizontal_grid)
    delta_grid = projected_horizontal_grid - projected_vertical_grid
    base_grid = projected_vertical_grid
    prev_rho_arc_points = prev_rho_arc.points()
    prev_mu_arc_points = prev_mu_arc.points()[1:] #removes the shell_point so it is not duplicated
    prev_arc_points = numpy.concatenate(prev_rho_arc_points, prev_mu_arc_points)
    projected_prev_arc_points = project(prev_arc_points)
    
    #optimize performance of the surface by finding the right mix of points to minimize focal error in both directions
    projected_screen_point = project(screen_point)
    #TODO: will be along the ending planes,checking how well that focuses for shell and screen point
    x_error_rays = blah
    y_error_rays = blah
    def error_for_weighting(weight):
        """
        Given a weighting between vertical and horizontal ribs, returns the error for the surface.
        Error is calculated as max of x and y error.
        Error is about how well rays along the extreme edges are focused to the screen point.
        """
        #make weighted average of each point in the grid
        grid = base_grid + weight * delta_grid
        
        #TODO: add greater weights to these points
        #include the points from existing arcs and shell point
        all_points = numpy.vstack(grid)
        all_points.concatenate(projected_prev_arc_points)
        
        #fit taylor poly to the points
        #TODO: need to make sure this is extremely accurate
        #if it isn't, then we have to ensure that we're using the taylor poly and casting rays to create the other edge arcs instead
        #because the point grid won't have enough accuracy
        poly = optics.localpoly.fit_poly(space, all_points)
        
        #cast rays against the taylor poly to measure x and y error
        x_error = _measure_error(poly, projected_screen_point, x_error_rays)
        y_error = _measure_error(poly, projected_screen_point, y_error_rays)
        error = max((x_error, y_error))
        print("%s: %s" % (weight, error))
        return error
    num_iterations = 20
    tolerance = 0.0001
    best_weight = scipy.optimize.fminbound(error_for_weighting, 0.0, 1.0, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    
    #finish initializing the patch
    grid = vertical_grid + best_weight * (horizontal_grid - vertical_grid)
    #TODO: unsure of the order here
    patch.grid = add_column(add_row(grid, prev_rho_arc_points), prev_mu_arc_points)
    
    return patch

#TODO: be explicit about which functions can be called before the thing is fully initialized
#actually, probably best to simply move those functions out completely
#TODO: pretty sure I can delete prev_rho_arc and prev_mu_arc and have all arcs be totally implicit (from the grid)
class Patch(object):
    def __init__(self, shell_point, screen_point, pre_rho_arc, prev_mu_arc, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane):
        self.shell_point = shell_point
        self.screen_point = screen_point
        self.pre_rho_arc = pre_rho_arc
        self.prev_mu_arc = prev_mu_arc
        self.rho_start_plane = rho_start_plane
        self.rho_end_plane = rho_end_plane
        self.mu_start_plane = mu_start_plane
        self.mu_end_plane = mu_end_plane
        
        #calculated or loaded
        self.grid = None
        
    def make_grid(self, arc_along_mu, num_slices=10):
        
        #create a bunch of arcs along the correct direction (in patch space)
        projected_eye_point = blah
        projected_shell_point = blah
        projected_screen_point = blah
        projected_plane = blah
        previous_normal_function = blah
        falloff = blah
        if arc_along_mu:
            arc_slice_angles = numpy.linspace(self.rho_start_plane.angle, self.rho_end_plane.angle, num_slices)[1:]
            cross_arc_slice_angles = numpy.linspace(self.mu_start_plane.angle, self.mu_end_plane.angle, num_slices)[1:]
        else:
            arc_slice_angles = numpy.linspace(self.mu_start_plane.angle, self.mu_end_plane.angle, num_slices)[1:]
            cross_arc_slice_angles = numpy.linspace(self.rho_start_plane.angle, self.rho_end_plane.angle, num_slices)[1:]
        cross_planes = [optics.arcplane.ArcPlane() for blah in blah]
        #does not contain the start points (eg, starting arcs)
        grid = numpy.zeros((num_slices, num_slices))
        for i in range(0, len(arc_slice_angles)):
            angle = arc_slice_angles[i]
            if arc_along_mu:
                arc_plane = optics.arcplane.ArcPlane(rho=angle)
            else:
                arc_plane = optics.arcplane.ArcPlane(mu=angle)
            #TODO: keep everything in real world coordinates. no projection except for taylor poly / grid
            arc = grow_arc(projected_eye_point, projected_shell_point, projected_screen_point, projected_plane, projected_end_plane, previous_normal_function, falloff)
            #intersect each of the cross planes with the arc to get actual points
            for j in range(0, len(cross_planes)):
                cross_plane = cross_planes[j]
                intersection = arc.intersect_poly(cross_plane)
                if arc_along_mu:
                    mu = i
                    rho = j
                else:
                    mu = j
                    rho = i
                grid[i][j] = intersection
        
        return grid
    
    def get_edge_arc(self, through_mu, direction):
        """
        :param through_mu: if True, we are arcing through different mu values (in a rho plane), else through different rho values (in a mu plane)
        :param direction: whether we are increasing or decreasing in angle
        """
        return blah
    
    def get_corner(self, mu, rho):
        """
        :returns: the point from this patch at the intersection of the planes closest to those mu and rho values
        """
        return blah
    
    def reflect_rays_no_bounds(self, rays):
        """
        :returns: the rays that have been reflected off of our taylor poly surface.
        Note: these rays will are not restricted to bounce only off of the in-domain section of the poly
        This is to prevent weirdness at the edges
        """
        return reflected_rays
    
    
    

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
        mu_start_plane,
        mu_end_plane,
        rho_start_plane,
        rho_end_plane,
        prev_mu_patch=None,
        prev_rho_patch=None
        ):
    """
    Given a bunch of initial parameters, create a new part of the surface.
    This is an initialization function for Patch
    """
    
    poly_order = optics.globals.POLY_ORDER
    
    #figure out which direction we are growing (in the mu rho directions)
    mu_direction = FORWARD
    if mu_start_plane.angle < mu_end_plane.angle:
        mu_direction = BACKWARD
    rho_direction = FORWARD
    if rho_start_plane.angle < rho_end_plane.angle:
        rho_direction = BACKWARD
    
    #either use the previous (neighboring) arc, or create it if there is no patch there because this is the zero edge
    #do in both directions
    if prev_mu_patch != None:
        prev_mu_arc = prev_mu_patch.get_edge_arc(False, mu_direction)
        prev_mu_normal_function = prev_mu_patch.surface_normal_function()
    else:
        prev_mu_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
    if prev_rho_patch != None:
        prev_rho_arc = prev_rho_patch.get_edge_arc(True, rho_direction)
        prev_rho_normal_function = prev_rho_patch.surface_normal_function()
    else:
        prev_rho_arc = optics.arc.new_grow_arc(shell_point, screen_point, mu_start_plane, rho_start_plane, rho_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)

    #create a grid of points for two conflicting sets of ribs
    vertical_grid = _make_grid(True, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, prev_mu_normal_function)
    horizontal_grid = _make_grid(False, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, prev_rho_normal_function)
    
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
    prev_rho_arc_points = prev_rho_arc.points
    prev_mu_arc_points = prev_mu_arc.points[1:] #removes the shell_point so it is not duplicated
    prev_arc_points = numpy.concatenate(prev_rho_arc_points, prev_mu_arc_points)
    projected_prev_arc_points = project(prev_arc_points)
    
    #optimize performance of the surface by finding the right mix of points to minimize focal error in both directions
    taylor_surface_radius = 1.1 * max((
        numpy.linalg.norm(prev_mu_arc.points[0] - prev_mu_arc.points[-1]),
        numpy.linalg.norm(prev_rho_arc.points[0] - prev_rho_arc.points[-1])
    ))
    projected_screen_point = project(screen_point)
    projected_ray_normal = space.normal_to_space(normalize(shell_point - ORIGIN))
    polys = {}
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
        poly = optics.localpoly.fit_poly(space, all_points, projected_screen_point, taylor_surface_radius, poly_order)
        #save it for later so we don't have to recalculate
        polys[weight] = poly
        
        #cast rays against the taylor poly to measure x and y error
        projected_x_error_rays = [Ray(grid[i][0] - projected_ray_normal, grid[i][0]) for i in range(0, len(grid))]
        projected_y_error_rays = [Ray(grid[0][i] - projected_ray_normal, grid[0][i]) for i in range(0, len(grid[0]))]
        x_error = _measure_error(poly, projected_screen_point, projected_x_error_rays)
        y_error = _measure_error(poly, projected_screen_point, projected_y_error_rays)
        error = max((x_error, y_error))
        print("%s: %s" % (weight, error))
        return error
    num_iterations = 20
    tolerance = 0.0001
    best_weight = scipy.optimize.fminbound(error_for_weighting, 0.0, 1.0, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    
    #finish initializing the patch
    grid = vertical_grid + best_weight * (horizontal_grid - vertical_grid)
    partial_grid = numpy.hstack((grid, prev_rho_arc_points))
    full_grid = numpy.vstack((partial_grid, prev_mu_arc_points))
    #TODO: unsure if that is right...
    x = blah
    patch = Patch(shell_point, screen_point, grid, polys[best_weight], space, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane)
    
    return patch

def _make_grid(arc_along_mu, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, previous_normal_function, num_slices=10):
        
    #create a bunch of arcs along the correct direction (in patch space)
    if arc_along_mu:
        arc_slice_angles = numpy.linspace(rho_start_plane.angle, rho_end_plane.angle, num_slices)[1:]
        start_plane = mu_start_plane
        end_plane = mu_end_plane
    else:
        arc_slice_angles = numpy.linspace(mu_start_plane.angle, mu_end_plane.angle, num_slices)[1:]
        start_plane = rho_start_plane
        end_plane = rho_end_plane
    #does not contain the start points (eg, starting arcs)
    grid = numpy.zeros((num_slices, num_slices))
    for i in range(0, len(arc_slice_angles)):
        angle = arc_slice_angles[i]
        if arc_along_mu:
            arc_plane = optics.arcplane.ArcPlane(rho=angle)
        else:
            arc_plane = optics.arcplane.ArcPlane(mu=angle)
        arc = optics.arc.new_grow_arc(ORIGIN, shell_point, screen_point, arc_plane, start_plane, end_plane, previous_normal_function)
        for j in range(1, len(arc)):
            if arc_along_mu:
                mu = i
                rho = j
            else:
                mu = j
                rho = i
            grid[mu][rho] = arc[i]
    return grid

def _measure_error(poly, screen_point, rays):
    """
    Figures out where the poly reflects the rays, and returns the distance from there to the screen point
    """
    reflected_rays = poly.reflect_rays(rays)
    distance = 0.0
    for ray in reflected_rays:
        distance += math.sqrt(distToLineSquared(screen_point, ray.start, ray.end))
    return distance / len (rays)

class Patch(object):
    """
    A grid of points defined at fixed angular distance between each.
    Includes the start and end arcs (eg, those points that lie on the start and end mu and rho planes).
    Is the result of some computation to define the best shape
    """
    
    def __init__(self, shell_point, screen_point, grid, poly, poly_space, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane):
        self.shell_point = shell_point
        self.screen_point = screen_point
        self.grid = grid
        self.poly = poly
        self.poly_space = poly_space
        self.rho_start_plane = rho_start_plane
        self.rho_end_plane = rho_end_plane
        self.mu_start_plane = mu_start_plane
        self.mu_end_plane = mu_end_plane
    
    def get_edge_arc(self, want_arc_in_mu, direction):
        """
        :param want_arc_in_mu: if True, we want an arc with constant mu, else we want constant rho
        :param direction: whether we are increasing or decreasing in angle
        """
        if want_arc_in_mu:
            if direction == FORWARD:
                return self.grid[:, len(self.grid)-1]
            else:
                return self.grid[:, 0]
        else:
            if direction == FORWARD:
                return self.grid[0]
            else:
                return self.grid[-1]
    
    def get_corner(self, mu, rho):
        """
        :returns: the point from this patch at the intersection of the planes closest to those mu and rho values
        """
        if math.fabs(self.mu_start_plane.angle - mu) < math.fabs(self.mu_end_plane.angle - mu):
            mu_idx = 0
        else:
            mu_idx = len(self.grid)-1
        if math.fabs(self.rho_start_plane.angle - rho) < math.fabs(self.rho_end_plane.angle - rho):
            rho_idx = 0
        else:
            rho_idx = len(self.grid)-1
        return self.grid[mu_idx][rho_idx]
    
    def reflect_rays_no_bounds(self, rays):
        """
        :returns: the rays that have been reflected off of our taylor poly surface.
        Note: these rays will are not restricted to bounce only off of the in-domain section of the patch
        This is to prevent weirdness at the edges, but you should be careful to ignore reflections that are outside of the patch space
        """
        projected_rays = [Ray(self.poly_space.point_to_space(ray.start), self.poly_space.point_to_space(ray.end)) for ray in rays]
        reflected_rays = self.poly.reflect_rays(projected_rays)
        return [Ray(self.poly_space.point_from_space(ray.start), self.poly_space.point_from_space(ray.end)) for ray in reflected_rays]
    
    def surface_normal_function(self):
        """
        :returns: a function that defines the surface normal vector field for this patch
        :rtype: function(Point3D) -> Point3D
        """
        #use the vector field to define the exact shape of the arc
        desired_light_direction = -1.0 * normalize(self.shell_point)
        def surface_normal(point):
            """
            Defines the surface normal at each point.
            """
            point_to_screen_vec = normalize(self.screen_point - point)
            surface_normal = normalize(point_to_screen_vec + desired_light_direction)
            return surface_normal
        return surface_normal
    

import scipy.integrate
import scipy.optimize
import matplotlib.pyplot

from optics.base import * # pylint: disable=W0401,W0614
import optics.globals
import optics.debug
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
    
    new_horizontal_arcing(
        shell_point,
        screen_point,
        mu_start_plane,
        mu_end_plane,
        rho_start_plane,
        rho_end_plane,
        poly_order
        )
    
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
        prev_mu_normal_function = None
    if prev_rho_patch != None:
        prev_rho_arc = prev_rho_patch.get_edge_arc(True, rho_direction)
        prev_rho_normal_function = prev_rho_patch.surface_normal_function()
    else:
        prev_rho_arc = optics.arc.new_grow_arc(shell_point, screen_point, mu_start_plane, rho_start_plane, rho_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
        prev_rho_normal_function = None
        
    understand_arcs(shell_point,
        screen_point,
        mu_start_plane,
        mu_end_plane,
        rho_start_plane,
        rho_end_plane,
        prev_mu_arc,
        prev_rho_arc,
        poly_order
        )
        
    #create a grid of points for two conflicting sets of ribs
    vertical_grid = _make_grid(True, prev_rho_arc, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, prev_mu_normal_function)
    horizontal_grid = _make_grid(False, prev_mu_arc, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, prev_rho_normal_function)
    
    if optics.debug.RIB_CREATION:
        #plot the original points and the resulting interpolated points
        axes = matplotlib.pyplot.subplot(111, projection='3d')
        vertical_points = numpy.vstack(vertical_grid)
        horizontal_points = numpy.vstack(horizontal_grid)
        axes.scatter(vertical_points[:, 0], vertical_points[:, 1], vertical_points[:, 2], c='r', marker='o').set_label('vertical grid')
        axes.scatter(horizontal_points[:, 0], horizontal_points[:, 1], horizontal_points[:, 2], c='g', marker='o').set_label('horizontal grid')
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
    
    #convert the grids to local space for this patch so we can fit taylor polys
    shell_point_to_eye_normal = normalize(ORIGIN - shell_point)
    shell_point_to_screen_normal = normalize(screen_point - shell_point)
    shell_point_surface_normal = normalize(shell_point_to_eye_normal + shell_point_to_screen_normal)
    space = CoordinateSpace(shell_point, shell_point_surface_normal)
    #TODO: This could be done MUCH more efficiently...
    def project(grid):
        """
        Convert grids into a different space. 
        """
        new_grid = numpy.zeros((len(grid), len(grid[0]), 3))
        for i in range(0, len(grid)):
            for j in range(0, len(grid[0])):
                new_grid[i][j] = space.point_to_space(grid[i][j])
        return new_grid
    projected_vertical_grid = project(vertical_grid)
    projected_horizontal_grid = project(horizontal_grid)
    delta_grid = projected_horizontal_grid - projected_vertical_grid
    base_grid = projected_vertical_grid
    prev_rho_arc_points = prev_rho_arc.points
    prev_mu_arc_points = prev_mu_arc.points[1:] #removes the shell_point so it is not duplicated
    prev_arc_points = numpy.vstack((prev_rho_arc_points, prev_mu_arc_points))
    projected_prev_arc_points = numpy.array([space.point_to_space(point) for point in prev_arc_points])
    
    #optimize performance of the surface by finding the right mix of points to minimize focal error in both directions
    taylor_surface_radius = 1.1 * max((
        numpy.linalg.norm(prev_mu_arc.points[0] - prev_mu_arc.points[-1]),
        numpy.linalg.norm(prev_rho_arc.points[0] - prev_rho_arc.points[-1])
    ))
    projected_screen_point = space.point_to_space(screen_point)
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
        all_points = numpy.vstack((numpy.vstack(grid), projected_prev_arc_points))
        
        #fit taylor poly to the points
        #TODO: need to make sure this is extremely accurate
        #if it isn't, then we have to ensure that we're using the taylor poly and casting rays to create the other edge arcs instead
        #because the point grid won't have enough accuracy
        poly = optics.localpoly.fit_poly(space, all_points, projected_screen_point, taylor_surface_radius, poly_order)
        #save it for later so we don't have to recalculate
        polys[weight] = poly
        
        if optics.debug.TAYLOR_SURFACE_CREATION:
            #plot the original points and the resulting interpolated points
            axes = matplotlib.pyplot.subplot(111, projection='3d')
            base_points = numpy.vstack(projected_vertical_grid)
            delta_points = numpy.vstack(projected_horizontal_grid)
            axes.scatter(base_points[:, 0], base_points[:, 1], base_points[:, 2], c='r', marker='o').set_label('base grid')
            axes.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='g', marker='o').set_label('weighted grid')
            x, y = all_points[:, 0], all_points[:, 1]
            wireframe = axes.plot_wireframe(x, y, poly.get_z_for_plot(x, y))
            wireframe.set_label("surface")
            wireframe.set_color("g")
            axes.scatter(delta_points[:, 0], delta_points[:, 1], delta_points[:, 2], c='b', marker='o').set_label('delta points')
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
            matplotlib.pyplot.legend()
            matplotlib.pyplot.show()
        
        #cast rays against the taylor poly to measure x and y error
        projected_x_error_rays = [Ray(grid[i][0] - projected_ray_normal, grid[i][0]) for i in range(0, len(grid))]
        projected_y_error_rays = [Ray(grid[0][i] - projected_ray_normal, grid[0][i]) for i in range(0, len(grid[0]))]
        x_error = _measure_error(poly, projected_screen_point, projected_x_error_rays)
        y_error = _measure_error(poly, projected_screen_point, projected_y_error_rays)
        error = max((x_error, y_error))
        print("%s: %s (max of %s, %s)" % (weight, error, x_error, y_error))
        return error
    num_iterations = 4#20
    tolerance = 0.0001
    best_weight = scipy.optimize.fminbound(error_for_weighting, 0.0, 1.0, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    
    #finish initializing the patch
    grid = vertical_grid + best_weight * (horizontal_grid - vertical_grid)
    full_grid = numpy.zeros((len(grid)+1, len(grid)+1, 3))
    full_grid[1:,1:] = grid
    full_grid[:,0] = prev_rho_arc_points
    full_grid[0,1:] = prev_mu_arc_points
    patch = Patch(shell_point, screen_point, grid, polys[best_weight], space, rho_start_plane, rho_end_plane, mu_start_plane, mu_end_plane)
    
    return patch

def _make_grid(arc_along_mu, prev_arc, shell_point, screen_point, mu_start_plane, mu_end_plane, rho_start_plane, rho_end_plane, previous_normal_function):
        
    #create a bunch of arcs along the correct direction (in patch space)
    if arc_along_mu:
        arc_slice_angles = numpy.linspace(rho_start_plane.angle, rho_end_plane.angle, optics.globals.NUM_SLICES)[1:]
        start_plane = mu_start_plane
        end_plane = mu_end_plane
    else:
        arc_slice_angles = numpy.linspace(mu_start_plane.angle, mu_end_plane.angle, optics.globals.NUM_SLICES)[1:]
        start_plane = rho_start_plane
        end_plane = rho_end_plane
    #does not contain the start points (eg, starting arcs)
    grid = numpy.zeros((optics.globals.NUM_SLICES-1, optics.globals.NUM_SLICES-1, 3))
    debug_points = []
    for i in range(0, len(arc_slice_angles)):
        angle = arc_slice_angles[i]
        shell_point_from_arc = prev_arc.points[i+1]
        if arc_along_mu:
            arc_plane = optics.arcplane.ArcPlane(rho=angle)
        else:
            arc_plane = optics.arcplane.ArcPlane(mu=angle)
        arc = optics.arc.new_grow_arc(shell_point_from_arc, screen_point, arc_plane, start_plane, end_plane, previous_normal_function)
        for j in range(1, len(arc.points)):
            if arc_along_mu:
                mu = i
                rho = j-1
            else:
                mu = j-1
                rho = i
            grid[mu][rho] = arc.points[j]
            
        if optics.debug.INDIVIDUAL_RIB_CREATION:
            debug_points += list(arc.points)
            #plot the original points and the resulting interpolated points
            axes = matplotlib.pyplot.subplot(111, projection='3d')
            points = numpy.array(debug_points)
            axes.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o').set_label('grid')
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')
            matplotlib.pyplot.legend()
            matplotlib.pyplot.show()
    return grid

def _measure_error(poly, screen_point, rays):
    """
    Figures out where the poly reflects the rays, and returns the distance from there to the screen point
    """
    reflected_rays = poly._local_reflect_rays(rays)
    distance = 0.0
    for ray in reflected_rays:
        distance += math.sqrt(distToLineSquared(screen_point, ray.start, ray.end))
    if optics.debug.TAYLOR_SURFACE_FOCAL_ERROR:
        #TODO: screen is really hard to find.
        axes = matplotlib.pyplot.subplot(111, projection='3d')
        size = 5
        num_points = 10
        x, y = numpy.meshgrid(numpy.linspace(-size, size, num_points), numpy.linspace(-size, size, num_points))
        axes.scatter(x, y, poly.get_z_for_plot(x, y), c='r', marker='o').set_label('patch')
        for ray in reflected_rays:
            debug_dist = numpy.linalg.norm(screen_point - ray.start)
            scaled_ray = Ray(ray.start, debug_dist * normalize(ray.end-ray.start) + ray.start)
            axes.plot([scaled_ray.start[0], scaled_ray.end[0]], [scaled_ray.start[1], scaled_ray.end[1]], [scaled_ray.start[2], scaled_ray.end[2]], label="ray")
        axes.plot([screen_point[0]], [screen_point[1]], [screen_point[2]], label="screen")
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
    #TODO: make a spot diagram as well so I can see distortion
        
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
        reflected_rays = self.poly._local_reflect_rays(projected_rays)
        result = []
        for ray in reflected_rays:
            if ray != None:
                result.append(Ray(self.poly_space.point_from_space(ray.start), self.poly_space.point_from_space(ray.end)))
            else:
                result.append(None)
        return result
    
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
    
def understand_arcs(
        shell_point,
        screen_point,
        mu_start_plane,
        mu_end_plane,
        rho_start_plane,
        rho_end_plane,
        prev_mu_arc,
        prev_rho_arc,
        poly_order
        ):
    #decide where existing arcs focus
    mu_end_focal_point = calculate_next_focal_point_from_arc(prev_mu_arc, screen_point)
    
    #debug: let's see what happens if we re-create the arc and consider the focus for this new point
    #hhmmm...  didn't work but I don't really understand why
    #let's try walking halfway, then halfway
    
    first_half_mu_end_plane = optics.arcplane.ArcPlane(mu=mu_end_plane.mu/2.0)
    first_half = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, first_half_mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
    first_half_focal_point = calculate_next_focal_point_from_arc(first_half, screen_point)
    second_half = optics.arc.new_grow_arc(first_half.points[0], first_half_focal_point, rho_start_plane, first_half_mu_end_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
    points = list(first_half.points)
    points.pop()
    first_half.points = numpy.array(points + list(second_half.points))
    
    mu_end_focal_point = calculate_next_focal_point_from_arc(first_half, screen_point, draw=True)
    
    #def original_surface_normal_function(point, t=None):
    #    point_to_screen_vec = normalize(screen_point - point)
    #    surface_normal = normalize(point_to_screen_vec - normalize(shell_point))
    #    return surface_normal
    #outer_shell_point = prev_mu_arc.points[-1]
    #def outer_point_surface_normal_function(point, t=None):
    #    point_to_screen_vec = normalize(mu_end_focal_point - point)
    #    surface_normal = normalize(point_to_screen_vec - normalize(outer_shell_point))
    #    return surface_normal
    #def surface_normal_function(point, t=None):
    #    dist_to_start = numpy.linalg.norm(point - shell_point)
    #    dist_to_end = numpy.linalg.norm(point - outer_shell_point)
    #    total_dist = dist_to_start + dist_to_end
    #    orig = original_surface_normal_function(point)
    #    new = outer_point_surface_normal_function(point)
    #    return normalize((dist_to_end / total_dist) * orig + (dist_to_start / total_dist) * new)
    #
    #prev_mu_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order, surface_normal_function=surface_normal_function)
    #mu_end_focal_point = calculate_next_focal_point_from_arc(prev_mu_arc, screen_point, surface_normal_function, draw=True)
    check_performance(prev_mu_arc, surface_normal_function)
    #end debug
    
    rho_end_focal_point = calculate_next_focal_point_from_arc(prev_rho_arc, screen_point)
    
    #grow new arcs
    next_mu_arc = optics.arc.new_grow_arc(prev_rho_arc.points[-1], rho_end_focal_point, rho_end_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
    next_rho_arc = optics.arc.new_grow_arc(prev_mu_arc.points[-1], mu_end_focal_point, mu_end_plane, rho_start_plane, rho_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order)
    
    #plot them all
    print("%s (%s -> %s)" % (numpy.linalg.norm(next_mu_arc.points[-1] - next_rho_arc.points[-1]), next_rho_arc.points[-1], next_mu_arc.points[-1]))
    axes = matplotlib.pyplot.subplot(111, projection='3d')
    axes.scatter(prev_mu_arc.points[:, 0], prev_mu_arc.points[:, 1], prev_mu_arc.points[:, 2], c='darkred', marker='o').set_label('grid')
    axes.scatter(prev_rho_arc.points[:, 0], prev_rho_arc.points[:, 1], prev_rho_arc.points[:, 2], c='darkblue', marker='o').set_label('grid')
    axes.scatter(next_mu_arc.points[:, 0], next_mu_arc.points[:, 1], next_mu_arc.points[:, 2], c='red', marker='o').set_label('grid')
    axes.scatter(next_rho_arc.points[:, 0], next_rho_arc.points[:, 1], next_rho_arc.points[:, 2], c='blue', marker='o').set_label('grid')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    
    #decide where the new arcs focus
    calculate_next_focal_point_from_arc(blah)
    calculate_next_focal_point_from_arc(blah)
    
    #see how far off those points are
    #if it's reasonable, make a taylor surface and check its focus
    
#TODO: this is all copy-pasted from somewhere. refactor if we actually use this code.
def calculate_next_focal_point_from_arc(arc, screen_point, surface_normal_function=None, draw=False):
    
    LIGHT_RADIUS = 3.0
    num_rays = 10
    
    #TODO: put me back
    #create a bunch of parallel rays coming from the eye going towards the end of the arc
    ray_vector = normalize(arc.points[-1])
    #ray_vector = normalize(arc.points[0])
    
    #figure out where reflections off of each arc point would go
    desired_light_direction = -1.0 * normalize(arc.points[0])
    def get_surface_normal(point):
        """
        Defines the surface normal at each point.
        """
        point_to_screen_vec = normalize(screen_point - point)
        surface_normal = normalize(point_to_screen_vec + desired_light_direction)
        return surface_normal
    if surface_normal_function == None:
        surface_normal_function = get_surface_normal
    reflected_rays = []
    for arc_point in arc.points:
        normal = surface_normal_function(arc_point)
        reverse_ray_direction = -1.0 * ray_vector
        midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
        reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
        reflection = arc_point + reflection_direction
        reflected_rays.append(Ray(arc_point, reflection))
    
    approximate_screen_normal = sum([normalize(ray.start - ray.end) for ray in reflected_rays]) / len(reflected_rays)
    def calculate_spot_size(distance):
        """
        :returns: average distance from the central point for the plane at this distance
        """
        screen_plane = Plane(distance * approximate_screen_normal * -1.0 + arc.points[-1], approximate_screen_normal)
        points = []
        for ray in reflected_rays:
            points.append(screen_plane.intersect_line(ray.start, ray.end))
        average_point = sum(points) / len(points)
        errors = [numpy.linalg.norm(p - average_point) for p in points]
        if draw:
            #keep a fixed scale to x and y so that each graph can be compared with the previous
            #should probably print the errors as well
            print errors
            print sum(errors) / len(errors)
            
            #use coordinate space to move everything to the xy plane
            space = CoordinateSpace(screen_plane._point, screen_plane._normal, straighten_cross_vector(Point3D(1.0, 0.0, 0.0), screen_plane._normal))
            transformed_points = numpy.array([space.point_to_space(p) for p in points])
            matplotlib.pyplot.plot(transformed_points[:, 0], transformed_points[:, 1], "r", linestyle='None', marker='o', label="rays at %s" % (distance))
            matplotlib.pyplot.legend()
            matplotlib.pyplot.show()
            
        print sum(errors) / len(errors)
        return sum(errors) / len(errors)
    previous_distance = numpy.linalg.norm(arc.points[0] - screen_point)
    min_dist = previous_distance * 0.9
    max_dist = previous_distance * 1.1
    num_iterations = 20
    tolerance = 0.0001
    
    #if draw:
    #    for d in numpy.linspace(72.0, 73.0, 11):
    #        calculate_spot_size(d)
    
    
    best_dist = scipy.optimize.fminbound(calculate_spot_size, min_dist, max_dist, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    focal_point = best_dist * approximate_screen_normal * -1.0 + arc.points[-1]
    print "done. focal point is %s away from the original" % (numpy.linalg.norm(screen_point - focal_point))
    return focal_point

def mu_from_point(point):
    return -1.0 * math.atan(point[0] / -point[2])

def calculate_focal_point_from_points(points, ray_direction, normal_function, previous_distance, center_shell_point, draw=False):
    reflected_rays = reflect_rays_from_points(points, ray_direction, normal_function)
    
    approximate_screen_normal = sum([normalize(ray.start - ray.end) for ray in reflected_rays]) / len(reflected_rays)
    best_points = {}
    def calculate_spot_size(distance):
        """
        :returns: average distance from the central point for the plane at this distance
        """
        plane_point = distance * approximate_screen_normal * -1.0 + center_shell_point
        screen_plane = Plane(plane_point, approximate_screen_normal)
        points = []
        for ray in reflected_rays:
            points.append(screen_plane.intersect_line(ray.start, ray.end))
        average_point = sum(points) / len(points)
        #print "%s: %s" % (distance, average_point)
        errors = [numpy.linalg.norm(p - average_point) for p in points]
        if draw:
            #keep a fixed scale to x and y so that each graph can be compared with the previous
            #should probably print the errors as well
            print errors
            print sum(errors) / len(errors)
            
            #use coordinate space to move everything to the xy plane
            space = CoordinateSpace(screen_plane._point, screen_plane._normal, straighten_cross_vector(Point3D(1.0, 0.0, 0.0), screen_plane._normal))
            transformed_points = numpy.array([space.point_to_space(p) for p in points])
            matplotlib.pyplot.plot(transformed_points[:, 0], transformed_points[:, 1], "r", linestyle='None', marker='o', label="rays at %s" % (distance))
            matplotlib.pyplot.legend()
            matplotlib.pyplot.show()
        final_error = sum(errors) / len(errors)
        best_points[distance] = average_point
        #print final_error
        return final_error
    min_dist = previous_distance * 0.9
    max_dist = previous_distance * 1.1
    num_iterations = 20
    tolerance = 0.0001
    
    #if draw:
    #    for d in numpy.linspace(72.0, 73.0, 11):
    #        calculate_spot_size(d)
    
    
    best_dist = scipy.optimize.fminbound(calculate_spot_size, min_dist, max_dist, maxfun=num_iterations, xtol=tolerance, full_output=False, disp=0)
    focal_point = best_points[best_dist]
    #print "final: %s: %s" % (best_dist, focal_point)
    #print "done. focal point is %s away from the original" % (numpy.linalg.norm(screen_point - focal_point))
    
    return focal_point

def reflect_rays_from_points(points, ray_vector, surface_normal_function):
    reflected_rays = []
    for arc_point in points:
        normal = surface_normal_function(arc_point)
        reverse_ray_direction = -1.0 * ray_vector
        midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
        reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
        reflection = arc_point + reflection_direction
        reflected_rays.append(Ray(arc_point, reflection))
    return reflected_rays

def measure_performance(reflected_rays, screen_point):
    distance = 0.0
    for ray in reflected_rays:
        distance += math.sqrt(distToLineSquared(screen_point, ray.start, ray.end))
    return distance / len(reflected_rays)

def new_horizontal_arcing(
        shell_point,
        screen_point,
        mu_start_plane,
        mu_end_plane,
        rho_start_plane,
        rho_end_plane,
        poly_order
        ):
    num_slices = 32
    #prev_mu_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order, num_slices=num_slices)
    angle_step = mu_end_plane.angle
    while mu_end_plane.angle < math.pi / 3.8:
    
        previous_distance = numpy.linalg.norm(shell_point - screen_point)
        
        new_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order, num_slices=num_slices)
        
        #for lots of little points along it, with very small bounds around the area, see where it focuses on the screen
        #thus we define a mu -> screen location function for the "native" focus
        def original_normal_function(point, t=None):
            point_to_screen_vec = normalize(screen_point - point)
            surface_normal = normalize(point_to_screen_vec - normalize(shell_point))
            return surface_normal
        
        reflected_rays = reflect_rays_from_points(new_arc.points, normalize(new_arc.points[-1]), original_normal_function)
        focal_point =  calculate_focal_point_from_points(new_arc.points, normalize(new_arc.points[-1]), original_normal_function, previous_distance, new_arc.points[-1])
        print("%s error: %s" % (mu_end_plane.angle, measure_performance(reflected_rays, focal_point)))
        
        shell_point = new_arc.points[-1]
        screen_point = focal_point
        mu_start_plane = mu_end_plane
        mu_end_plane = optics.arcplane.ArcPlane(mu=mu_end_plane.mu+angle_step)
    
    #mapping = [(-0.0000001, screen_point)]
    #for i in range(1, len(prev_mu_arc.points)-1):
    #    focal_point = calculate_focal_point_from_points([prev_mu_arc.points[i+k] for k in (-1, 0, 1)], normalize(prev_mu_arc.points[i]), original_normal_function, previous_distance, prev_mu_arc.points[i])
    #    mu = mu_from_point(prev_mu_arc.points[i])
    #    mapping.append((mu, focal_point))
    ##have to extend the mapping to the end so that it works out nicely
    #next_mu = (mapping[-1][0] - mapping[-2][0]) + mapping[-1][0]
    #next_point = (mapping[-1][1] - mapping[-2][1]) + mapping[-1][1]
    #mapping.append((next_mu+0.00001, next_point))
    #
    ##then re-integrate the original arc, moving light from those screen locations out directly to the eye
    #def better_surface_normals(point, t=None):
    #    mu = mu_from_point(point)
    #    mu_screen_point = None
    #    for i in range(0, len(mapping)-1):
    #        if mu > mapping[i][0] and mu < mapping[i+1][0]:
    #            screen_start = mapping[i][1]
    #            screen_vector = mapping[i+1][1] - screen_start
    #            percentage_along = (mu - mapping[i][0]) / (mapping[i+1][0] - mapping[i][0])
    #            mu_screen_point = percentage_along * screen_vector + screen_start
    #            break
    #    if mu_screen_point == None:
    #        #assert False
    #        mu_screen_point = mapping[-1][1]
    #    desired_light_direction = -1.0 * normalize(point)
    #    point_to_screen_vec = normalize(mu_screen_point - point)
    #    surface_normal = normalize(point_to_screen_vec + desired_light_direction)
    #    return surface_normal
    #new_arc = optics.arc.new_grow_arc(shell_point, screen_point, rho_start_plane, mu_start_plane, mu_end_plane, previous_normal_function=None, falloff=-1.0, poly_order=poly_order, surface_normal_function=better_surface_normals, num_slices=num_slices)

    #axes = matplotlib.pyplot.subplot(111, projection='3d')
    #axes.scatter(prev_mu_arc.points[:, 0], prev_mu_arc.points[:, 1], prev_mu_arc.points[:, 2], c='darkred', marker='o').set_label('grid')
    #axes.scatter(new_arc.points[:, 0], new_arc.points[:, 1], new_arc.points[:, 2], c='darkblue', marker='o').set_label('grid')
    #for pointset, color, norm in ((prev_mu_arc.points, 'red', original_normal_function), (new_arc.points, 'blue', better_surface_normals)):
    #    for point in pointset:
    #        points = numpy.array([point, point + 40.0 * norm(point)])
    #        axes.plot(points[:, 0], points[:, 1], points[:, 2], c=color)
    #axes.set_xlabel('X')
    #axes.set_ylabel('Y')
    #axes.set_zlabel('Z')
    #matplotlib.pyplot.show()

    #then characterize new performance around each of the points
    #note: for small regions around each of the points, the performance should by definition be really good
    #question is what happens to performance as we increase the size of those performance testing bounds
    
    print "done"
    
    ##first characterize performance for the start point:
    ##reflect all the rays
    ##measure difference from screen point
    #reflected_rays = reflect_rays_from_points(new_arc.points, normalize(new_arc.points[0]), better_surface_normals)
    #print("error: %s" % (measure_performance(reflected_rays, screen_point)))
    #
    ##then characterize performance for the final point
    ##reflect all rays and calculate focal point
    ##measure difference from focal point
    #reflected_rays = reflect_rays_from_points(new_arc.points, normalize(new_arc.points[-1]), better_surface_normals)
    #focal_point =  calculate_focal_point_from_points(new_arc.points, normalize(new_arc.points[-1]), better_surface_normals, previous_distance, new_arc.points[-1])
    #print("error: %s" % (measure_performance(reflected_rays, focal_point)))
    #
    ##do the same for the mid point
    #mid_point = new_arc.points[len(new_arc.points) / 2]
    #reflected_rays = reflect_rays_from_points(new_arc.points, normalize(mid_point), better_surface_normals)
    #focal_point =  calculate_focal_point_from_points(new_arc.points, normalize(mid_point), better_surface_normals, previous_distance, mid_point)
    #print("error: %s" % (measure_performance(reflected_rays, focal_point)))
    
    x = 4
    
    

import math

import numpy
import scipy.integrate
import scipy.optimize

import matplotlib.pyplot

from optics.base import * # pylint: disable=W0401,W0614
import optics.debug
import optics.globals
import optics.utils
import optics.parallel
import optics.scale
import optics.rotation_matrix

#TODO: use previous_normal_function and falloff
def new_grow_arc(
    shell_point,
    screen_point,
    arc_plane,
    start_cross_plane,
    end_cross_plane,
    previous_normal_function=None,
    falloff=-1.0,
    step_size=0.01,
    poly_order=None,
    surface_normal_function=None
    ):
    """
    Creates an arc through space within the given plane, ending at the end plane. The arc is
    created by integrating through a vector field that is defined by the shell point and screen
    point.
    
    :param shell_point: where this arc will begin
    :type  shell_point: Point2D
    :param screen_point: where this arc should focus light
    :type  screen_point: Point2D
    :param arc_plane: the plane that will contain this arc. note that it is orthogonal to end_arc_plane.
    :type  arc_plane: ArcPlane
    :param start_cross_plane: where this arc begins
    :type  start_cross_plane: ArcPlane
    :param end_cross_plane: where this arc should terminate
    :type  end_cross_plane: ArcPlane
    :param previous_normal_function: maps from points in the previous patch space to surface normal.
        Used for smoothing the transition between the two patches.
    :type  previous_normal_function: function(Point3D) -> Point3D
    :param falloff: over what proportion of the range should the correction for the previous screen point fall off.
        Controls how flat the surface will be at the starting point. Higher numbers will make the surface flatter
    :type  falloff: float
    :param step_size: how accurately we should integrate through the vector field. is roughly in mm
    :type  step_size: float
    :param poly_order: what order of polynomial to fit for the arc
    :type  poly_order: int
    
    :returns: the arc that best focuses light to that screen_point
    :rtype: NewArc
    """
    
    if poly_order == None:
        poly_order = optics.globals.POLY_ORDER

    #use the vector field to define the exact shape of the arc
    desired_light_direction = -1.0 * normalize(shell_point)
    def vector_field_derivative(point, t):
        """
        Defines the derivative through the vector field.
        """
        point_to_screen_vec = normalize(screen_point - point)
        if surface_normal_function == None:
            surface_normal = normalize(point_to_screen_vec + desired_light_direction)
        else:
            surface_normal = surface_normal_function(point, t)
        derivative = normalize(numpy.cross(surface_normal, arc_plane.normal))
        return derivative
    #note: since arcs are mostly linear, we calculate the max_t value based on how far we're travelling, roughly
    max_t = 1.3 * numpy.linalg.norm(end_cross_plane.project(shell_point) - shell_point)
    t_values = numpy.arange(0.0, max_t, step_size)
    points = scipy.integrate.odeint(vector_field_derivative, shell_point, t_values)
    
    #fit a 2D polynomial to the points
    direction = 1.0
    if end_cross_plane.angle < 0.0:
        direction = -1.0
    arc_poly = create_arc_poly(arc_plane, direction, shell_point, screen_point, points, poly_order)
    
    #find the intersection between that poly and each cross plane
    cross_arc_slice_angles = numpy.linspace(start_cross_plane.angle, end_cross_plane.angle, optics.globals.NUM_SLICES)[1:]
    if start_cross_plane.mu != None:
        cross_planes = [optics.arcplane.ArcPlane(mu=angle) for angle in cross_arc_slice_angles]
    else:
        cross_planes = [optics.arcplane.ArcPlane(rho=angle) for angle in cross_arc_slice_angles]
    arc_points = [shell_point]
    for cross_plane in cross_planes:
        intersection = arc_poly.intersect_plane(cross_plane)
        assert intersection != None, "failed to intersect the plane. Maybe need to increase max_t?"
        arc_points.append(intersection)
    arc_points = numpy.array(arc_points)
    arc = NewArc(arc_points)
    
    if optics.debug.ARC_CREATION:
        #plot the original points and the resulting interpolated points
        axes = matplotlib.pyplot.subplot(111, projection='3d')
        axes.plot(points[:, 0], points[:, 1], points[:, 2], c='r', marker='x', label='original points')
        axes.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2], c='g', marker='o', label='arc points')
        axes.plot([ORIGIN[0]], [ORIGIN[1]], [ORIGIN[2]], c='b', marker='*', label='eye')
        axes.plot([screen_point[0]], [screen_point[1]], [screen_point[2]], c='r', marker='*', label='screen')
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
    
    return arc

#TODO: put a ton of weight on 0,0, must pass through there?
#TODO: possibly put less weight on points that are farther away?
def create_arc_poly(arc_plane, direction, shell_point, screen_point, points, poly_order):
    """
    Constructs and initializes an ArcPoly
    """
    
    #transform the points into a local space that will actually fit a polynomial
    shell_point_in_plane = arc_plane.world_to_local(shell_point)
    screen_point_in_plane = arc_plane.world_to_local(screen_point)
    arc_poly = ArcPoly(arc_plane, shell_point_in_plane, screen_point_in_plane, direction)
    projected_points = numpy.array([arc_poly.plane_to_local(arc_plane.world_to_local(point)) for point in points])
    fudge = 0.1
    arc_poly.min_x = numpy.min(projected_points[:, 0]) - fudge
    arc_poly.max_x = numpy.max(projected_points[:, 0]) + fudge
    
    #fit the polynomial to the points
    coefficients = numpy.polynomial.polynomial.polyfit(projected_points[:, 0], projected_points[:, 1], poly_order)
    poly = numpy.polynomial.polynomial.Polynomial(coefficients) # pylint: disable=E1101
    
    if optics.debug.POLYARC_FITTING:
        projected_origin = arc_poly.plane_to_local(arc_plane.world_to_local(ORIGIN))
        projected_screen_point = arc_poly.plane_to_local(arc_plane.world_to_local(screen_point))
        matplotlib.pyplot.plot(projected_points[:, 0], projected_points[:, 1], "r", label="arc points")
        x = numpy.linspace(arc_poly.min_x, arc_poly.max_x, 200)
        matplotlib.pyplot.plot(x, poly(x), "g", label="polynomial")
        matplotlib.pyplot.plot(projected_origin[0], projected_origin[1], "ro", label="eye")
        matplotlib.pyplot.plot(projected_screen_point[0], projected_screen_point[1], "bo", label="screen point")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
    
    arc_poly._poly = poly
    arc_poly._derivative = poly.deriv(1)
    
    return arc_poly

class ArcPoly(object):
    """
    A polynomial in an ArcPlane.
    
    Has its own internal crazy coordinate system.
    """
    
    def __init__(self, arc_plane, shell_point, screen_point, direction):
        self.arc_plane = arc_plane
        self.shell_point = shell_point
        self.screen_point = screen_point
        self.direction = direction
        
        self._local_to_plane_translation = shell_point
        self._plane_to_local_translation = -1.0 * self._local_to_plane_translation
        
        shell_to_eye_normal = -1.0 * normalize(shell_point)
        angle = normalized_vector_angle(shell_to_eye_normal, Point2D(0.0, 1.0))
        self._local_to_plane_rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self._plane_to_local_rotation = numpy.linalg.inv(self._local_to_plane_rotation)
        #TODO: this is dumb.
        if math.fabs(self._plane_to_local_rotation.dot(shell_to_eye_normal)[1] - 1.0) > 0.00000001:
            angle *= -1.0
            self._local_to_plane_rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            self._plane_to_local_rotation = numpy.linalg.inv(self._local_to_plane_rotation)
        
        #set later
        self.min_x = None
        self.max_x = None
        self._poly = None
        self._derivative = None
    
    def plane_to_local(self, point):
        """
        Converts Point2D in plane space to Point2D in arc space
        """
        transformed_point = self._plane_to_local_rotation.dot(point + self._plane_to_local_translation)
        return Point2D(self.direction * transformed_point[0], transformed_point[1])
        
    def local_to_plane(self, point):
        """
        Converts Point2D in arc space to Point2D in plane space
        """
        reflected_point = Point2D(self.direction * point[0], point[1])
        return self._local_to_plane_rotation.dot(reflected_point) + self._local_to_plane_translation
        
    def intersect_plane(self, plane):
        """
        Given an ArcPlane, return the world point where it intersects our polynomial
        """
        ray_end = self.plane_to_local(self.arc_plane.world_to_local(plane.view_normal))
        ray_start = self.plane_to_local(Point2D(0.0, 0.0))
        ray = Ray(ray_start, ray_end)
        
        #convert ray to the form mx + b
        line_poly = _convert_line_to_poly_coefs(ray.start, ray.end)
        
        #TODO: this is a bit odd. should probably select the point closer to the start..
        #select the intersection that is within the x range, and closest to zero, and mostly real (because of numerical inaccuracies)
        roots = numpy.polynomial.polynomial.polyroots(self._poly.coef - line_poly)
        
        if optics.debug.POLYARC_INTERSECTIONS:
            ray_list = numpy.array([ray.start, ray.end])
            matplotlib.pyplot.plot(ray_list[:, 0], ray_list[:, 1], "r-", label="ray")
            x = numpy.linspace(self.min_x, self.max_x, 100)
            matplotlib.pyplot.plot(x, self._poly(x), "b", label="poly")
            matplotlib.pyplot.plot(x, numpy.polynomial.polynomial.Polynomial(self._poly.coef - line_poly)(x), "g", label="difference")
            projected_origin = self.plane_to_local(self.arc_plane.world_to_local(Point3D(0.0, 0.0, 0.0)))
            matplotlib.pyplot.plot(projected_origin[0], projected_origin[1], "ro", label="eye")
            matplotlib.pyplot.plot(self.screen_point[0], self.screen_point[1], "bo", label="screen")
            matplotlib.pyplot.legend()
            print roots
            matplotlib.pyplot.show()
        
        best_root = float("inf")
        for root in numpy.real_if_close(roots, tol=1000000):
            if numpy.isreal(root):
                root = numpy.real(root)
                if root > self.min_x and root < self.max_x:
                    if math.fabs(root) < math.fabs(best_root):
                        best_root = root
        if best_root == float("inf"):
            return None
        return self.arc_plane.local_to_world(self.local_to_plane(Point2D(best_root, self._poly(best_root))))

#TODO: replace Arc() with this class completely, as soon as this new approach works...
class NewArc(object):
    """
    An Arc is simply a list of points.
    All points are contained within a single ArcPlane (eg, mu or rho is constant).
    All points are an equal angular step apart in either mu or rho (eg, whichever is not constant).
    All points are in world space.
    This makes meshing, rendering, etc, extremely easy.
    """
    
    def __init__(self, points):
        self.points = points

#TODO: implement flatness_falloff. make a function that defines the derivative given some parameters, then use both of them for the real derivative function
#OPT: make these derivative functions in cython instead
def grow_arc(shell_point, screen_point, screen_normal, prev_screen_point, arc_plane, end_arc_plane, step_size=0.01, flatness_falloff=0.2):
    """
    Creates an arc.
    
    :param shell_point: where this arc will begin
    :type  shell_point: Point2D
    :param screen_point: where this arc should focus light
    :type  screen_point: Point2D
    :param screen_normal: normal to the screen. only used in the case where FORCE_FLAT_SCREEN is True.
    :type  screen_normal: Point2D
    :param prev_screen_point: where the previous arc focused light
    :type  prev_screen_point: Point2D
    :param arc_plane: the plane that will contain this arc. note that it is orthogonal to end_arc_plane. all points are in this space.
    :type  arc_plane: ArcPlane
    :param end_arc_plane: where this arc should terminate
    :type  end_arc_plane: ArcPlane
    :param step_size: how accurately we should integrate through the vector field. is roughly in mm
    :type  step_size: float
    :param flatness_falloff: over what proportion of the range should the correction for the previous screen point fall off.
    Controls how flat the surface will be at the starting point. Higher numbers will make the surface flatter
    :type  flatness_falloff: float
    
    :returns: the arc that best focuses light to that screen_point
    :rtype: Arc
    """
    
    print("%s %s" % (end_arc_plane.mu, arc_plane.rho))
    
    #create the arc with basic parameters. not fully initialized yet, but can at least use the transformation
    direction = 1.0
    if end_arc_plane.angle < 0.0:
        direction = -1.0
    arc = Arc(arc_plane, shell_point, screen_point, screen_normal, direction)
    
    #transform everything into the arc's coordinates
    projected_screen_point = arc._plane_to_local(screen_point)
    projected_shell_point = arc._plane_to_local(shell_point)
    projected_prev_creen_point = arc._plane_to_local(prev_screen_point)
    projected_origin = arc._plane_to_local(arc_plane.world_to_local(Point3D(0.0, 0.0, 0.0)))
    projected_end_plane_line_start = projected_origin
    projected_end_plane_line_end = arc._plane_to_local(arc_plane.world_to_local(end_arc_plane.view_normal))
    
    #define the derivative
    desired_light_direction_off_screen_towards_eye = normalize(projected_origin - projected_shell_point)
    def f(point, t):
        point_to_screen_vec = normalize(projected_screen_point - point)
        surface_normal = normalize(point_to_screen_vec + desired_light_direction_off_screen_towards_eye)
        derivative = Point2D(surface_normal[1], -surface_normal[0])
        return derivative
    
    #TODO: this is a pretty arbitrary, pointlessly high max_t
    #we can calulate the max_t much more precisely because arcs are mostly linear...
    
    #use the vector field to define the exact shape of the arc
    max_t = 2.0 * optics.globals.LIGHT_RADIUS
    t_values = numpy.arange(0.0, max_t, step_size)
    points = scipy.integrate.odeint(f, projected_shell_point, t_values)
    
    #fit a 2D polynomial to the points
    #TODO: put a ton of weight on 0,0, must pass through there
    #TODO: possibly put less weight on points that are farther away?
    #TODO: also, we don't really care that much about points that have gone farther than this poly really will..  should probably trim those
    #which should be relatively easy, because it's going to be relatively flat and we know how far approximately we'll go
    coefficients = numpy.polynomial.polynomial.polyfit(points[:, 0], points[:, 1], optics.globals.POLY_ORDER)
    arc._poly = numpy.polynomial.polynomial.Polynomial(coefficients)
    arc._derivative = arc._poly.deriv(1)
    
    #plt.plot(points[:,0], points[:, 1],"r")
    #plt.plot(points[:,0], arc._poly(points[:, 0]),"b")
    #plt.plot(projected_origin[0], projected_origin[1],"ro")
    #plt.plot(projected_screen_point[0], projected_screen_point[1],"bo")
    #plt.show()
    
    #intersect the poly and the projected_end_plane_vector to find the bounds for the arc
    #find the first intersection that is positive and closest to 0
    line_poly = _convert_line_to_poly_coefs(projected_end_plane_line_start, projected_end_plane_line_end)
    roots = numpy.real(numpy.polynomial.polynomial.polyroots(coefficients - line_poly))
    positive_roots = [r for r in roots if r > 0]
    
    #if end_arc_plane.mu > 0.5 and arc_plane.rho > 0.14:
    #if arc_plane.rho > 0.14:
    #if arc_plane.rho != None:
        #plt.plot(points[:,0], points[:, 1],"r")
        #plt.plot(points[:,0], arc._poly(points[:, 0]),"b")
        #plt.plot(points[:,0], numpy.polynomial.polynomial.Polynomial(line_poly)(points[:,0]), "g-")
        #plt.plot(projected_origin[0], projected_origin[1],"ro")
        #plt.plot(projected_screen_point[0], projected_screen_point[1],"bo")
        #plt.show()
    
    if len(positive_roots) <= 0:
        x_values = numpy.array([0.0, 80.0])
        plt.plot(points[:,0], points[:, 1],"r")
        plt.plot(x_values, b + m * x_values, 'b-')
        plt.plot(projected_origin[0], projected_origin[1], "ro")
        plt.plot(projected_screen_point[0], projected_screen_point[1], "bo")
        plt.show()
    
    arc.max_x = min(positive_roots)
    
    return arc

def _convert_line_to_poly_coefs(start, end):
    n = end - start
    m = n[1] / n[0]
    b = end[1] - m * end[0]
    return [b, m] + (optics.globals.POLY_ORDER - 1) * [0.0]

#TODO: doc which parameters are in which coordinate systems.
class Arc(object):
    """
    Shell point is redefined to be 0,0
    The screen point is on the +y axis
    The arc extends between x=0 and x=something
    
    All input and output to this function should be in ArcPlane coordinates
    """
    
    def __init__(self, arc_plane, shell_point, screen_point, screen_normal, direction):
        self.arc_plane = arc_plane
        self.shell_point = shell_point
        self.start_point = shell_point #just an alias
        self.screen_point = screen_point
        #note: only used in the case where FORCE_FLAT_SCREEN is True. Otherwise the screen normal is not really defined
        self.screen_normal = screen_normal
        self.direction = direction
        
        self._local_to_plane_translation = shell_point
        self._plane_to_local_translation = -1.0 * self._local_to_plane_translation
        
        shell_to_screen_normal = normalize(screen_point - shell_point)
        angle = normalized_vector_angle(shell_to_screen_normal, Point2D(0.0, 1.0))
        self._local_to_plane_rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self._plane_to_local_rotation = numpy.linalg.inv(self._local_to_plane_rotation)
        #TODO: this is dumb.
        if math.fabs(self._plane_to_local_rotation.dot(shell_to_screen_normal)[1] - 1.0) > 0.00000001:
            angle *= -1.0
            self._local_to_plane_rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            self._plane_to_local_rotation = numpy.linalg.inv(self._local_to_plane_rotation)
        
        #these need to be set later, after figuring out the polynomial
        self._poly = None
        self._derivative = None
        
        #hacks
        self.render_rays = []
    
    def _plane_to_local(self, point):
        """
        Converts Point2D in plane space to Point2D in arc space
        """
        transformed_point = self._plane_to_local_rotation.dot(point + self._plane_to_local_translation)
        return Point2D(self.direction * transformed_point[0], transformed_point[1])
        
    def _local_to_plane(self, point):
        """
        Converts Point2D in arc space to Point2D in plane space
        """
        reflected_point = Point2D(self.direction * point[0], point[1])
        return self._local_to_plane_rotation.dot(reflected_point) + self._local_to_plane_translation
    
    def fast_normal(self, point):
        """
        Just checks for the derivative at x
        """
        transformed_point = self._plane_to_local(point)
        local_normal = rotate_90(normalize(Point2D(1.0, self._derivative(transformed_point[0]))))
        reflected_point = Point2D(self.direction * local_normal[0], local_normal[1])
        return self._local_to_plane_rotation.dot(reflected_point)
    
    def _debug_plot_intersection(self, ray):
        #convert ray to the form mx + b
        line_poly = _convert_line_to_poly_coefs(self._plane_to_local(ray.start), self._plane_to_local(ray.end))
        roots = numpy.polynomial.polynomial.polyroots(self._poly.coef - line_poly)
        
        ray_list = numpy.array([self._plane_to_local(ray.start), self._plane_to_local(ray.end)])
        plt.plot(ray_list[:, 0], ray_list[:, 1], "r-")
        x = numpy.linspace(0.0, self.max_x, 100)
        plt.plot(x, self._poly(x),"b")
        plt.plot(x, numpy.polynomial.polynomial.Polynomial(self._poly.coef - line_poly)(x),"g")
        projected_screen_point = self._plane_to_local(self.screen_point)
        projected_origin = self._plane_to_local(self.arc_plane.world_to_local(Point3D(0.0, 0.0, 0.0)))
        plt.plot(projected_origin[0], projected_origin[1],"ro")
        plt.plot(projected_screen_point[0], projected_screen_point[1],"bo")
        print roots
        plt.show()
        
    def fast_arc_plane_intersection(self, ray):
        """
        Note: collides with this as a line, not a ray (eg, the directionality is ignored)
        Also assumes that there is just one intersection in the start -> end range
        
        Also, this obviously works with rays that are 2D only, eg, must be in our arc plane
        """
        #convert ray to the form mx + b
        line_poly = _convert_line_to_poly_coefs(self._plane_to_local(ray.start), self._plane_to_local(ray.end))
        roots = numpy.polynomial.polynomial.polyroots(self._poly.coef - line_poly)
        
        for root in numpy.real_if_close(roots, tol=1000000):
            if numpy.isreal(root):
                root = numpy.real(root)
                if root > 0 and root < self.max_x:
                    return self._local_to_plane(Point2D(root, self._poly(root)))
        return None
    
    def fast_arc_plane_reflection(self, ray):
        intersection = self.fast_arc_plane_intersection(ray)
        if intersection != None:
            normal = self.fast_normal(intersection)
            reverse_ray_direction = normalize(ray.start - ray.end)
            midpoint = closestPointOnLine(reverse_ray_direction, Point2D(0.0, 0.0), normal)
            reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
            return intersection, reflection_direction
        return None, None
    
    def draw_rays(self, rays):
        for ray in rays:
            ray_list = numpy.array([self._plane_to_local(ray.start), self._plane_to_local(ray.end)])
            plt.plot(ray_list[:, 0], ray_list[:, 1], "r-")
        x = numpy.linspace(0.0, self.max_x, 100)
        plt.plot(x, self._poly(x),"b")
        plt.show()
    
    @property
    def end_point(self):
        return self._local_to_plane(Point2D(self.max_x, self._poly(self.max_x)))
    
    @property
    def world_end_point(self):
        return self.arc_plane.local_to_world(self.end_point)
    
    @property
    def world_screen_point(self):
        return self.arc_plane.local_to_world(self.screen_point)
    
    @property
    def world_shell_point(self):
        return self.arc_plane.local_to_world(self.shell_point)
    
    
    
    
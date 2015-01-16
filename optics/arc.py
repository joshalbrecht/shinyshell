
import numpy

from optics.base import *
import optics.globals
import optics.rotation_matrix

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
    
    #create the arc with basic parameters. not fully initialized yet, but can at least use the transformation
    direction = 1.0
    if end_arc_plane.angle < 0.0:
        direction = -1.0
    arc = Arc(arc_plane, shell_point, screen_point, screen_normal, direction)
    
    #transform everything into the arc's coordinates
    projected_screen_point = arc._plane_to_local(screen_point)
    projected_shell_point = arc._plane_to_local(shell_point)
    projected_sprev_creen_point = arc._plane_to_local(prev_screen_point)
    projected_end_plane_vector = arc._plane_to_local(end_arc_plane.view_normal)
    
    #define the derivative
    angle_vec = get_angle_vec_from_point(shell_point)
    desired_light_direction_off_screen_towards_eye = normalize(arc_plane.project(-1.0 * angle_vector_to_vector(angle_vec, PRINCIPAL_RAY)))
    def f(point, t):
        point_to_screen_vec = normalize(projected_screen_point - point)
        surface_normal = normalize(point_to_screen_vec + desired_light_direction_off_screen_towards_eye)
        return Point2D(-surface_normal[1], surface_normal[0])
    
    #use the vector field to define the exact shape of the arc
    max_t = 1.1 * light_radius
    t_values = numpy.arange(0.0, max_t, step_size)
    points = scipy.integrate.odeint(f, projected_shell_point, t_values)
    
    #fit a 2D polynomial to the points
    #TODO: put a ton of weight on 0,0, must pass through there
    #TODO: possibly put less weight on points that are farther away?
    #TODO: also, we don't really care that much about points that have gone farther than this poly really will..  should probably trim those
    #which should be relatively easy, because it's going to be relatively flat and we know how far approximately we'll go
    weights = blah
    coefficients = numpy.polyfit(points[:, 0], points[:, 1], optics.globals.POLY_ORDER)
    arc._poly = np.polynomial.polynomial.Polynomial(coefficients)
    
    #intersect the poly and the projected_end_plane_vector to find the bounds for the arc
    #find the first intersection that is positive and closest to 0
    roots = np.polynomial.polynomial.polyroots(poly_coeff - [99, -1, 0])
    arc.end_point = arc.intersection(projected_end_plane_ray)
    
    return arc

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
        self.screen_point = screen_point
        #note: only used in the case where FORCE_FLAT_SCREEN is True. Otherwise the screen normal is not really defined
        self.screen_normal = screen_normal
        self.direction = direction
        
        self._local_to_plane_translation = shell_point
        self._plane_to_local_translation = -1.0 * self._local_to_plane_translation
        
        shell_to_screen_normal = normalize(screen_point - shell_point)
        angle = normalized_vector_angle(shell_to_screen_normal, Point2D(0.0, 1.0))
        self._local_to_plane_rotation = numpy.array([math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)])
        self._plane_to_local_rotation = numpy.linalg.inv(self._local_to_plane_rotation)
        
        #these need to be set later, after figuring out the polynomial
        self._poly = None
        self.end_point = None
    
    def _plane_to_local(self, point):
        """
        Converts Point2D in plane space to Point2D in arc space
        """
        transformed_point = self._plane_to_local_rotation.dot(reflected_point + self._plane_to_local_translation)
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
        return blah(point[0])
    
    def fast_arc_plane_intersection(self, plane_ray):
        """
        Note: collides with this as a line, not a ray (eg, the directionality is ignored)
        Also assumes that there is just one intersection in the start -> end range
        
        Also, this obviously works with rays that are 2D only, eg, must be in our arc plane
        """
        #convert ray to the form mx + b
        m = blah
        b = blah
        roots = numpy.polynomial.polynomial.polyroots(self._poly.blah - [b, m, 0])
        #find the intersection that is between start and end, and closest to the start of the line
        return blah
    
    
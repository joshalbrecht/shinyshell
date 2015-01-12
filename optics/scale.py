
import numpy
import scipy.optimize

#this is the one thing that is allowed to import *
from optics.base import *
import optics.mesh

class PolyScale(object): 
    def __init__(   self,
                    shell_point=None,
                    pixel_point=None,
                    angle_vec=None,
                    poly=None,
                    world_to_local_rotation=None,
                    world_to_local_translation=None,
                    domain_cylinder_point=None,
                    domain_cylinder_radius=None,
                    screen_normal=None
                    ):
        assert shell_point != None
        assert pixel_point != None
        assert angle_vec != None
        self._shell_point = shell_point
        self._pixel_point = pixel_point
        self._angle_vec = angle_vec
        self._poly = poly
        self._world_to_local_rotation = world_to_local_rotation
        self._world_to_local_translation = world_to_local_translation
        self._domain_cylinder_point = domain_cylinder_point
        self._domain_cylinder_radius = domain_cylinder_radius
        self.screen_normal = screen_normal
        
        self.focal_error = -1.0
        self.shell_distance_error = None
        
        self._post_init()
        
    def _post_init(self):
        #arbitrary
        self._num_rays = 11
        self._pupil_radius = 3.0
        
        #derived
        self._local_to_world_rotation = numpy.linalg.inv(self._world_to_local_rotation)
        self._local_to_world_translation = -1.0 * self._world_to_local_translation
        
        #generated
        self._rays = None
        self._points = None
        self._mesh = None
        
    def __getstate__(self):
        state = {}
        state['_shell_point'] = self._shell_point
        state['_pixel_point'] = self._pixel_point
        state['_angle_vec'] = self._angle_vec
        state['_poly'] = self._poly
        state['_world_to_local_rotation'] = self._world_to_local_rotation
        state['_world_to_local_translation'] = self._world_to_local_translation
        state['_domain_cylinder_point'] = self._domain_cylinder_point
        state['_domain_cylinder_radius'] = self._domain_cylinder_radius
        state['focal_error'] = self.focal_error
        state['shell_distance_error'] = self.shell_distance_error
        state['screen_normal'] = self.screen_normal
        
        return state
    
    def __setstate__(self, state):
        self._shell_point = state['_shell_point']
        self._pixel_point = state['_pixel_point']
        self._angle_vec = state['_angle_vec']
        self._poly = state['_poly']
        self._world_to_local_rotation = state['_world_to_local_rotation']
        self._world_to_local_translation = state['_world_to_local_translation']
        self._domain_cylinder_point = state['_domain_cylinder_point']
        self._domain_cylinder_radius = state['_domain_cylinder_radius']
        self.focal_error = state['focal_error']
        self.shell_distance_error = state['shell_distance_error']
        self.screen_normal = state['screen_normal']
        
        self._post_init()

    @property
    def shell_point(self):
        return self._shell_point
    
    @property
    def pixel_point(self):
        return self._pixel_point
    
    @property
    def angle_vec(self):
        return self._angle_vec

    def set_scale_color(self, color):
        """
        :attr color: 4-item tuple of RGBA values
        """
        if self._mesh != None:
            self._mesh._color = color
    
    def points(self):
        if self._points == None:
            self._points = []
            points = self._poly.points()
            for p in points:
                self._points.append(self._local_to_world(p))
        return self._points
    
    def _arcs(self):
        arcs = self._poly.arcs()
        new_arcs = []
        for arc in arcs:
            new_arc = []
            for p in arc:
                new_arc.append(self._local_to_world(p))
            new_arcs.append(new_arc)
        return new_arcs
    
    def render(self):
        self.ensure_mesh()
        self._mesh.render()
        
        #if self._rays == None:
        #    self._calculate_rays()
        
        if self._rays != None:
            for ray in self._rays:
                ray.render()
                
    def _get_arc_and_start(self, ray_start, ray_direction, step_size):
        """
        walk along the ray, generating a bunch of points from the projection on to the scale
        """
        #transform the ray into scale coordinates
        transformed_ray_start = self._world_to_local(ray_start)
        transformed_ray_normal = self._world_to_local(ray_start + ray_direction) - transformed_ray_start
        
        #zero out the z portion of the ray and make a parameterized x/y line
        projected_normal = normalize(Point3D(transformed_ray_normal[0], transformed_ray_normal[1], 0.0))
        
        #if the start is out of domain, whatever. calculate and return that as the start point with no arc
        arc = []
        z = self._poly.eval_poly(transformed_ray_start[0], transformed_ray_start[1])
        point = Point3D(transformed_ray_start[0], transformed_ray_start[1], z)
        if not self._poly.in_domain(point):
            return self._local_to_world(point), arc
        arc.append(self._local_to_world(point))
        
        #otherwise, continue along the ray until you are no longer in domain
        while True:
            point += step_size * projected_normal
            point = Point3D(point[0], point[1], self._poly.eval_poly(point[0], point[1]))
            if not self._poly.in_domain(point):
                #start point is whatever point was the last one in the arc, basically
                return arc[-1], arc
            arc.append(self._local_to_world(point))
        
    def _get_shell_and_screen_point_from_ray(self, ray, screen_plane):
        """
        :returns: the position on the shell and screen where this ray would land, or None, None if it would not hit the scale
        """
        reflection_length = 1.1 * numpy.linalg.norm(self.shell_point - self.pixel_point)
        intersection, normal = self.intersection_plus_normal(ray.start, ray.end)
        if intersection == None:
            return None, None
        reverse_ray_direction = normalize(ray.start - ray.end)
        midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
        reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
        ray_to_screen = Ray(intersection, intersection + reflection_length * reflection_direction)
        plane_intersection = screen_plane.intersect_line(ray_to_screen.start, ray_to_screen.end)
        return intersection, plane_intersection
    
    def get_screen_points(self, rays, screen_plane):
        points = []
        for ray in rays:
            plane_intersection = self. _get_shell_and_screen_point_from_ray(ray, screen_plane)[1]
            if plane_intersection != None:
                points.append(plane_intersection)
        return points

    def ensure_mesh(self):
        if self._mesh == None:
            self._mesh = self._create_mesh()
        
    def _local_to_world(self, p):
        return self._local_to_world_rotation.dot(p) + self._local_to_world_translation
        
    def _world_to_local(self, p):
        return self._world_to_local_rotation.dot(p + self._world_to_local_translation)
    
    def _create_mesh(self):
        """
        Just makes an approximate mesh. For rendering mostly.
        """
        
        #make a mesh from those arcs
        base_mesh = optics.mesh.mesh_from_arcs(self._arcs())
        
        #trim the mesh given our domain cylinder
        trimmed_mesh = optics.mesh.trim_mesh_with_cone(base_mesh, Point3D(0.0, 0.0, 0.0), normalize(self._shell_point), self._domain_cylinder_radius)
        
        return optics.mesh.Mesh(trimmed_mesh)

    def intersection_plus_normal(self, start, end):
        """
        Just like intersection, but returns the normal as well
        """
        transformed_start = self._world_to_local(start)
        transformed_end = self._world_to_local(end)
        point = self._poly._intersection(transformed_start, normalize(transformed_end-transformed_start))
        if point == None:
            return None, None
        #calculate the normal as well
        normal = self._poly.normal(point)
        return self._local_to_world(point), self._local_to_world_rotation.dot(normal)
    
    def intersection(self, start, end):
        """
        Really the entire reason we switch to taylor poly instead. much better intersections hopefully...
        """
        transformed_start = self._world_to_local(start)
        transformed_end = self._world_to_local(end)
        point = self._poly._intersection(transformed_start, normalize(transformed_end-transformed_start))
        if point == None:
            return None
        return self._local_to_world(point)
    


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
        self._post_init()
        
    def _post_init(self):
        self.shell_distance_error = None
        self._num_rays = 11
        self._pupil_radius = 3.0
        self._rays = None
        self._local_to_world_rotation = numpy.linalg.inv(self._world_to_local_rotation)
        self._local_to_world_translation = -1.0 * self._world_to_local_translation
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
        return state
    
    def __setstate__(self, state):
        self._shell_point = state['_shell_point']
        self._pixel_point = state['_pixel_point']
        self._angle_vec = state['_angle_vec']
        self._poly = state['_poly']
        self._world_to_local_rotation = state['_world_to_local_rotation']
        self._poly = state['_world_to_local_translation']
        self._poly = state['_domain_cylinder_point']
        self._poly = state['_domain_cylinder_radius']
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
        
        if self._rays == None:
            self._calculate_rays()
        
        for ray in self._rays:
            ray.render()
            
    def _calculate_rays(self):
        infinite_rays = []
        base_eye_ray = Ray(Point3D(0,0,0), 100.0 * self._shell_point)
        if self._num_rays == 1:
            infinite_rays.append(base_eye_ray)
        else:
            for y in numpy.linspace(-self._pupil_radius, self._pupil_radius, num=self._num_rays):
                delta = Point3D(0, y, 0)
                infinite_rays.append(Ray(base_eye_ray.start + delta, base_eye_ray.end+delta))
        
        #just want to see how close we're getting to the correct pixel location:
        screen_plane = Plane(self._pixel_point, normalize(self.shell_point - self.pixel_point))
        distance_from_center = 0.0
        num_collisions = 0
        
        reflection_length = 1.1 * numpy.linalg.norm(self.shell_point - self.pixel_point)
        self._rays = []
        for ray in infinite_rays:
            intersection, normal = self.intersection_plus_normal(ray.start, ray.end)
            if intersection != None:
                self._rays.append(LightRay(ray.start, intersection))
                reverse_ray_direction = normalize(ray.start - ray.end)
                midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
                reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
                ray_to_screen = LightRay(intersection, intersection + reflection_length * reflection_direction)
                self._rays.append(ray_to_screen)
                
                plane_intersection = screen_plane.intersect_line(ray_to_screen.start, ray_to_screen.end)
                distance_from_center += numpy.linalg.norm(plane_intersection - self._pixel_point)
                num_collisions += 1
        self.focal_error = distance_from_center / num_collisions

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
        
        #TODO: rename the mesh module. it's too generic of a name
        #make a mesh from those arcs
        base_mesh = optics.mesh.mesh_from_arcs(self._arcs())
        
        #trim the mesh given our domain cylinder
        trimmed_mesh = optics.mesh.trim_mesh_with_cone(base_mesh, Point3D(0.0, 0.0, 0.0), normalize(self._shell_point), self._domain_cylinder_radius)
        
        return optics.mesh.Mesh(trimmed_mesh)

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
        if point == None:
            return None, None
        #calculate the normal as well
        normal = self._poly.normal(point)
        return self._local_to_world(point), self._local_to_world_rotation.dot(normal)
    
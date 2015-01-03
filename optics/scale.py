
import numpy
import scipy.optimize

#this is the one thing that is allowed to import *
from optics.base import *
import optics.mesh

#TODO: I think Scale should NOT be a child of Mesh. It should contain one though
#TODO: merge Scale and PolyScale classes

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
        base_eye_ray = Ray(Point3D(0,0,0), 100.0 * self._shell_point)
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
            intersection, normal = self.intersection_plus_normal(ray.start, ray.end)
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
                        world_to_local_translation=None,
                        domain_cylinder_point=None,
                        domain_cylinder_radius=None,
                        **kwargs):
        Scale.__init__(self, **kwargs)
        self._poly = poly
        self._world_to_local_rotation = world_to_local_rotation
        self._local_to_world_rotation = numpy.linalg.inv(world_to_local_rotation)
        self._world_to_local_translation = world_to_local_translation
        self._local_to_world_translation = -1.0 * world_to_local_translation
        self._domain_cylinder_point = domain_cylinder_point
        self._domain_cylinder_radius = domain_cylinder_radius
        self._points = None
        
    def render(self):
        if self._mesh == None:
            self.set_mesh(self._create_mesh())
        Scale.render(self)
        
    def _local_to_world(self, p):
        return self._local_to_world_rotation.dot(p) + self._local_to_world_translation
        
    def _world_to_local(self, p):
        return self._world_to_local_rotation.dot(p + self._world_to_local_translation)
    
    def points(self):
        if self._points == None:
            self._points = []
            
            cone_point = Point3D(0.0, 0.0, 0.0)
            cone_normal = normalize(self._shell_point)
            cone_radius = self._domain_cylinder_radius

            cone_end = cone_point + cone_normal*100.0
            sq_radius = cone_radius*cone_radius
            w = cone_point
            v = cone_end
            n = w - v
            v_w_sq_len = dist2(v, w)
            
            arcs = self._arcs()
            for arc in arcs:
                for p in arc:
                    delta = p - (v + (((p - v).dot(n)) / v_w_sq_len) * n)
                    if delta.dot(delta) < sq_radius:
                        self._points.append(p)
                
        return self._points
    
    def _arcs(self):
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
        return arcs
    
    def _create_mesh(self):
        """
        Just makes an approximate mesh. For rendering mostly.
        """
        
        #TODO: rename the mesh module. it's too generic of a name
        #make a mesh from those arcs
        base_mesh = optics.mesh.mesh_from_arcs(self._arcs())
        
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
        if point == None:
            return None, None
        #calculate the normal as well
        normal = self._poly.normal(point)
        return self._local_to_world(point), self._local_to_world_rotation.dot(normal)
    
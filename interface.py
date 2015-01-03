#!/usr/bin/python

"""
Installation instructions: sudo pip install pyglet

This is a hacked together application to directly manipulate the surface in 2D.

All assumptions and coordinates are the same as in main.py. It's simply 2D because we're viewing a 3D scene from (zoom, 0, 0) looking at (0, 0, 0)

Usage instructions:

left click and drag to move pixels or shell sections
right click to make a new piece of shell (and paired pixel)
middle mouse drag to pan
middle mouse roll to zoom
"""

import math
import sys
import itertools
from time import time, sleep

from OpenGL import GL, GLU
import pyglet
from pyglet.gl import *
import pyglet.window.key
import numpy
import scipy.integrate
import scipy.optimize

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

import rotation_matrix
from optics import Point3D, _normalize, _normalized_vector_angle, _get_arc_plane_normal, angle_vector_to_vector, AngleVector, distToSegment, closestPointOnLine
import mesh

import pyximport; pyximport.install()
import hacks.taylor_poly

#TODO: use the pyglet codes here instead
LEFT_MOUSE_BUTTON_CODE = 1L
MIDDLE_MOUSE_BUTTON_CODE = 2L
RIGHT_MOUSE_BUTTON_CODE = 4L

#enable this to speed up development. Just cuts back on a lot of precision
LOW_QUALITY_MODE = False

class Plane(object):
    def __init__(self, point, normal):
        self._point = point
        self._normal = normal
        
    # intersection function
    def intersect_line(self, p0, p1, epsilon=0.0000001):
        """
        p0, p1: define the line    
        return a Vector or None (when the intersection can't be found).
        """
    
        u = p1 - p0
        w = p0 - self._point
        dot = self._normal.dot(u)
    
        if abs(dot) > epsilon:
            # the factor of the point between p0 -> p1 (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # otherwise:
            #  < 0.0: behind p0.
            #  > 1.0: infront of p1.
            fac = -self._normal.dot(w) / dot
            return p0 + u*fac
        else:
            # The segment is parallel to plane
            return None

class Ray(object):
    def __init__(self, start, end):
        self._start = start
        self._end = end
        
    @property
    def start(self): 
        return self._start
    
    @property
    def end(self): 
        return self._end
        
class VisibleLineSegment(Ray):
    def __init__(self, start, end, color=(1.0, 1.0, 1.0)):
        Ray.__init__(self, start, end)
        self._color = color
        
    def render(self):
        glBegin(GL_LINES)
        glColor3f(*self._color)
        glVertex3f(*self._start)
        glVertex3f(*self._end)
        glEnd()

class LightRay(VisibleLineSegment):
    def __init__(self, start, end):
        VisibleLineSegment.__init__(self, start, end, color=(0.5, 0.5, 0.5))

class SceneObject(object):
    """
    :attr pos: the position of the object in real coordinates (mm)
    :attr change_handler: will be called if the position changes
    """
    
    scene_objects = []
    
    def __init__(self, pos=None, color=(1.0, 1.0, 1.0), change_handler=None, radius=1.0):
        assert pos != None, "Must define a position for a SceneObject"
        self._pos = pos
        self._color = color
        self._change_handler = change_handler
        self.radius = radius
        self.scene_objects.append(self)
        
    def on_change(self):
        if self._change_handler:
            self._change_handler()
            
    def render(self):
        glColor3f(*self._color)
        
    def distance_to_ray(self, ray):
        """
        :returns: the min distance between self._pos and ray
        """
        return distToSegment(self._pos, ray.start, ray.end)
        
    @property
    def pos(self): 
        return self._pos

    @pos.setter
    def pos(self, value): 
        self._pos = value
        self.on_change()
        
    @classmethod
    def pick_object(cls, ray):
        """
        :returns: the thing that is closest to the ray, if anything was close enough for the bounding sphere to be intersected
        :rtype: SceneObject
        """
        best_dist = float("inf")
        best_obj = None
        for obj in cls.scene_objects:
            dist = obj.distance_to_ray(ray)
            if dist < obj.radius:
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj
        return best_obj

class ScreenPixel(SceneObject):
    def render(self):
        SceneObject.render(self)
        glBegin(GL_POINTS)
        glVertex3f(*self._pos)
        glEnd()

class ReflectiveSurface(SceneObject):
    def __init__(self, **kwargs):
        SceneObject.__init__(self, **kwargs)
        self.segments = []
        
    def render(self):
        SceneObject.render(self)
        for segment in self.segments:
            segment.render()
        
class ShellSection(object):
    """
    :attr shell: the object representing the center of the shell
    :attr pixel: the object representing the pixel on the screen
    :attr num_rays: the number of parallel rays emitted by the eye
    :attr pupil_radius: 1/2 of the width of the region that emits rays from the eye
    
    Note that when any of those attributes are set, the object will recalculate everything based on them
    """
    
    def __init__(self, shell_point, pixel_point, num_rays, pupil_radius):
        self._shell = ReflectiveSurface(pos=shell_point, color=(1.0, 1.0, 1.0), change_handler=self._on_change)
        self._pixel = ScreenPixel(pos=pixel_point, color=(1.0, 1.0, 1.0), change_handler=self._on_change)
        self._num_rays = num_rays
        self._pupil_radius = pupil_radius
        
        #calculated attributes
        self._rays = []
        
        self._dirty = True
        self._recalculate()
        
    def render(self):
        """Draw the shell section, pixel, and rays"""
        self._recalculate()
        
        self._shell.render()
        self._pixel.render()
        for ray in self._rays:
            ray.render()
            
    def _recalculate(self):
        """Update internal state when any of the interesting variables have changed"""
        
        ##note: this is just a hacky implementation right now to see if things are generally working
        #point_to_eye = _normalize(Point3D(0,0,0) - self._shell.pos)
        #point_to_pixel = _normalize(self._pixel.pos - self._shell.pos)
        #surface_normal = _normalize((point_to_eye + point_to_pixel) / 2.0)
        #tangent = Point3D(0.0, -1.0 * surface_normal[2], surface_normal[1])
        #start = self._shell.pos + tangent
        #end = self._shell.pos - tangent
        #segment = VisibleLineSegment(start, end)
        #self._shell.segments = [segment]
        #self._rays = [LightRay(Point3D(0,0,0), self._shell.pos), LightRay(self._shell.pos, self._pixel.pos)]
        
        if not self._dirty:
            return
        self._dirty = False
        
        principal_ray = Point3D(0.0, 0.0, -1.0)
        
        #figure out the angle of the primary ray
        phi = _normalized_vector_angle(principal_ray, _normalize(self._shell.pos))
        if self._shell.pos[1] < 0.0:
            theta = 3.0 * math.pi / 2.0
        else:
            theta = math.pi / 2.0
        
        shell_points = create_arc(principal_ray, self._shell.pos, self._pixel.pos, self.pupil_radius, AngleVector(theta, phi), is_horizontal=False)
        
        #create all of the segments
        segments = [VisibleLineSegment(shell_points[i-1], shell_points[i]) for i in range(1, len(shell_points))]
        
        #create all of the inifinite rays (those going from the eye to someplace really far away, in the correct direction)
        infinite_rays = []
        base_eye_ray = Ray(Point3D(0,0,0), 100.0 * self._shell.pos)
        if self._num_rays == 1:
            infinite_rays.append(base_eye_ray)
        else:
            for y in numpy.linspace(-self._pupil_radius, self.pupil_radius, num=self._num_rays):
                delta = Point3D(0, y, 0)
                infinite_rays.append(Ray(base_eye_ray.start + delta, base_eye_ray.end+delta))
        
        #NOTE: we are NOT actually reflecting these rays off of the surface right now.
        #'bounce' each ray off of the surface (eg, find the segment that it is closest to and use that as the termination)
        #this is used to create the rays that we will draw later
        #also remembers the earliest and latest segment index, so that we can drop everything not required for the surface
        earliest_segment_index = sys.maxint
        latest_segment_index = -1
        self._rays = []
        for ray in infinite_rays:
            best_index = -1
            best_sq_dist = float("inf")
            best_loc = None
            for i in range(0, len(segments)):
                seg = segments[i]
                tangent = _normalize(seg.end - seg.start)
                #janky rotation
                normal = Point3D(0, tangent[2], -tangent[1])
                plane = Plane(seg.start, normal)
                loc = plane.intersect_line(ray.start, ray.end)
                midpoint = (seg.start + seg.end) /2.0
                delta = midpoint - loc
                sq_dist = delta.dot(delta)
                if sq_dist < best_sq_dist:
                    best_index = i
                    best_loc = loc
                    best_sq_dist = sq_dist
            self._rays.append(LightRay(ray.start, best_loc))
            self._rays.append(LightRay(best_loc, self._pixel.pos))
            if best_index > latest_segment_index:
                latest_segment_index = best_index
            if best_index < earliest_segment_index:
                earliest_segment_index = best_index
        self._shell.segments = segments[earliest_segment_index:latest_segment_index]

    @property
    def num_rays(self): 
        return self._num_rays

    @num_rays.setter
    def num_rays(self, value): 
        self._num_rays = value
        self._on_change()
        
    @property
    def pupil_radius(self): 
        return self._pupil_radius

    @pupil_radius.setter
    def pupil_radius(self, value): 
        self._pupil_radius = value
        self._on_change()
        
    def _on_change(self):
        self._dirty = True
    

class Window(pyglet.window.Window):
    def __init__(self, refreshrate):
        super(Window, self).__init__(vsync = False)
        self.frames = 0
        self.framerate = pyglet.text.Label(text='Unknown', font_name='Verdana', font_size=8, x=10, y=10, color=(255,255,255,255))
        self.last = time()
        self.alive = 1
        self.refreshrate = refreshrate
        self.left_click = None
        self.middle_click = None
        
        self.selection = []
        self.zoom_multiplier = 1.2
        self.focal_point = Point3D(0.0, 20.0, 30.0)
        self.camera_point = Point3D(100.0, self.focal_point[1], self.focal_point[2])
        self.up_vector = Point3D(0.0, 1.0, 0.0)
        
        self.num_rays = 3
        self.pupil_radius = 2.0
        self.sections = []
        self.create_initial_sections()
        
        initial_shell_point = Point3D(0.0, 0.0, -60.0)
        initial_screen_point = Point3D(0.0, 40.0, -20.0)
        principal_ray = Point3D(0.0, 0.0, -1.0)
        self.scales = create_surface_via_scales(initial_shell_point, initial_screen_point, principal_ray)
        
    #TODO: someday can save and load sections as well perhaps, so we can resume after shutting down
    def create_initial_sections(self):
        """Initialize some semi-sensible sections"""
        self.sections = []
        self.sections.append(ShellSection(Point3D(0.0, 10.0, -70.0), Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
        #self.sections.append(ShellSection(Point3D(0.0, 0.0, -60.0), Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
        self.sections.append(ShellSection(Point3D(0.0, -5.0, -50.0), Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))

    def on_draw(self):
        self.render()
        
    def on_mouse_press(self, x, y, button, modifiers):
        if button == RIGHT_MOUSE_BUTTON_CODE:
            location = self._mouse_to_2d_plane(x, y)
            self.sections.append(ShellSection(location, Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
            
        elif button == LEFT_MOUSE_BUTTON_CODE:
            self.left_click = x,y
        
            ray = self._click_to_ray(x, y)
            obj = SceneObject.pick_object(ray)
            if obj:
                self.selection = [obj]
            else:
                self.selection = []
                
        elif button == MIDDLE_MOUSE_BUTTON_CODE:
            self.middle_click = x,y

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.left_click:
            for obj in self.selection:
                start_plane_location = self._mouse_to_2d_plane(x, y)
                end_plane_location = self._mouse_to_2d_plane(x+dx, y+dy)
                delta = end_plane_location - start_plane_location
                obj.pos += delta
        if self.middle_click:
            if modifiers & pyglet.window.key.MOD_SHIFT:
                self._rotate(dx, dy)
            else:
                start_plane_location = self._mouse_to_work_plane(x, y)
                end_plane_location = self._mouse_to_work_plane(x+dx, y+dy)
                delta = end_plane_location - start_plane_location
                self.focal_point += -1.0 * delta
                self.camera_point += -1.0 * delta
                
    def _rotate(self, dx, dy):
        angle_step = 0.01
        view_normal = _normalize(self.focal_point - self.camera_point)
        side_normal = numpy.cross(view_normal, self.up_vector)
        y_matrix = numpy.zeros((3,3))
        y_angle = dy * angle_step
        rotation_matrix.R_axis_angle(y_matrix, side_normal, y_angle)
        x_matrix = numpy.zeros((3,3))
        x_angle = -dx * angle_step
        rotation_matrix.R_axis_angle(x_matrix, self.up_vector, x_angle)
        matrix = x_matrix.dot(y_matrix)
        
        #translate up point and camera point to remove the focal_point offset, rotate them, then translate back
        camera_point = self.camera_point - self.focal_point
        up_point = camera_point + self.up_vector
        self.camera_point = matrix.dot(camera_point) + self.focal_point
        self.up_vector = _normalize((matrix.dot(up_point) + self.focal_point) - self.camera_point)
        
    def _mouse_to_work_plane(self, x, y):
        ray = self._click_to_ray(x, y)
        point = Plane(self.focal_point, _normalize(self.camera_point - self.focal_point)).intersect_line(ray.start, ray.end)
        return point
    
    def _mouse_to_2d_plane(self, x, y):
        ray = self._click_to_ray(x, y)
        point = Plane(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)).intersect_line(ray.start, ray.end)
        return point

    def on_mouse_release(self, x, y, button, modifiers):
        if button == LEFT_MOUSE_BUTTON_CODE:
            self.left_click = None
        elif button == MIDDLE_MOUSE_BUTTON_CODE:
            self.middle_click = None
            
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        distance = numpy.linalg.norm(self.camera_point - self.focal_point)
        for i in range(0, abs(scroll_y)):
            if scroll_y < 0:
                distance *= self.zoom_multiplier
            else:
                distance *= 1.0 / self.zoom_multiplier
        self.camera_point = self.focal_point + distance * _normalize(self.camera_point - self.focal_point)
        
    def _click_to_ray(self, x, y):
        model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        start = Point3D(*GLU.gluUnProject(x, y, 0.0, model=model, proj=proj, view=view))
        end = Point3D(*GLU.gluUnProject(x, y, 1.0, model=model, proj=proj, view=view))
        return Ray(start, end)

    def _draw_axis(self):
        axis_len = 10.0
        
        glBegin(GL_LINES)
        
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(axis_len, 0.0, 0.0)
        
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, axis_len, 0.0)
        
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, axis_len)
        
        glEnd()

    def render(self):
        
        self.clear()
        
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(self.camera_point[0], self.camera_point[1], self.camera_point[2],
                  self.focal_point[0], self.focal_point[1], self.focal_point[2],
                  self.up_vector[0], self.up_vector[1], self.up_vector[2])
        
        self._draw_axis()
        
        for section in self.sections:
            section.render()
            
        for scale in self.scales:
            scale.render()
        
        if time() - self.last >= 1:
            self.framerate.text = str(self.frames)
            self.frames = 0
            self.last = time()
        else:
            self.frames += 1
        self.framerate.draw()
        self.flip()
        
    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65, width / float(height), 0.01, 500)
        glMatrixMode(GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def on_close(self):
        self.alive = 0

    def run(self):
        while self.alive:
            self.render()
            # ----> Note: <----
            #  Without self.dispatch_events() the screen will freeze
            #  due to the fact that i don't call pyglet.app.run(),
            #  because i like to have the control when and what locks
            #  the application, since pyglet.app.run() is a locking call.
            event = self.dispatch_events()
            sleep(1.0/self.refreshrate)
            
class Scale(mesh.Mesh):
    def __init__(self, shell_point=None, pixel_point=None, angle_vec=None, **kwargs):
        mesh.Mesh.__init__(self, **kwargs)
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
        mesh.Mesh.render(self)
        
        if self._rays == None:
            self._calculate_rays()
        
        for ray in self._rays:
            ray.render()
            
    def _calculate_rays(self):
        infinite_rays = []
        base_eye_ray = Ray(Point3D(0,0,0), 1.2 * self._shell_point)
        if self._num_rays == 1:
            infinite_rays.append(base_eye_ray)
        else:
            for y in numpy.linspace(-self._pupil_radius, self._pupil_radius, num=self._num_rays):
                delta = Point3D(0, y, 0)
                infinite_rays.append(Ray(base_eye_ray.start + delta, base_eye_ray.end+delta))
        
        #TEMP: just want to see how close we're getting to the correct pixel location:
        screen_plane = Plane(self._pixel_point, _normalize(self.shell_point - self.pixel_point))
        
        reflection_length = 1.1 * numpy.linalg.norm(self.shell_point - self.pixel_point)
        self._rays = []
        for ray in infinite_rays:
            intersection, normal = self.intersection_plus_normal(ray.end, ray.start)
            normal *= -1.0
            if intersection != None:
                self._rays.append(LightRay(ray.start, intersection))
                reverse_ray_direction = _normalize(ray.start - ray.end)
                midpoint = closestPointOnLine(reverse_ray_direction, Point3D(0.0, 0.0, 0.0), normal)
                reflection_direction = (2.0 * (midpoint - reverse_ray_direction)) + reverse_ray_direction
                ray_to_screen = LightRay(intersection, intersection + reflection_length * reflection_direction)
                self._rays.append(ray_to_screen)
                
                plane_intersection = screen_plane.intersect_line(ray_to_screen.start, ray_to_screen.end)
                print numpy.linalg.norm(plane_intersection - self._pixel_point)
                
class PolyScale(Scale):
    def __init__(self,  poly=None,
                        world_to_local_rotation=None,
                        local_to_world_rotation=None,
                        world_to_local_translation=None,
                        domain_cylinder_point=None,
                        domain_cylinder_radius=None,
                        **kwargs):
        Scale.__init__(self, **kwargs)
        self._poly = poly
        self._world_to_local_rotation = world_to_local_rotation
        self._local_to_world_rotation = local_to_world_rotation
        self._world_to_local_translation = world_to_local_translation
        self._local_to_world_translation = -1.0 * world_to_local_translation
        self._domain_cylinder_point = domain_cylinder_point
        self._domain_cylinder_radius = domain_cylinder_radius
        
    def render(self):
        if self._mesh == None:
            self.set_mesh(self._create_mesh())
        Scale.render(self)
        
    def _local_to_world(self, p):
        return self._local_to_world_rotation.dot(p) + self._local_to_world_translation
        
    def _world_to_local(self, p):
        return self._world_to_local_rotation.dot(p + self._world_to_local_translation)
    
    def _create_mesh(self):
        """
        Just makes an approximate mesh. For rendering mostly.
        """
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
        
        #TODO: rename the mesh module. it's too generic of a name
        #make a mesh from those arcs
        base_mesh = mesh.mesh_from_arcs(arcs)
        
        #trim the mesh given our domain cylinder
        trimmed_mesh = mesh.trim_mesh_with_cone(base_mesh, Point3D(0.0, 0.0, 0.0), _normalize(self._shell_point), self._domain_cylinder_radius)
        
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
        point = self._poly._intersection(transformed_start, _normalize(transformed_end-transformed_start))
        #calculate the normal as well
        normal = -1.0 * self._poly.normal(point)
        return self._local_to_world(point), self._local_to_world_rotation.dot(normal)
    
def create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=None):
    assert is_horizontal != None, "Must pass this parameter"
    
    #define a vector field for the surface normals of the shell.
    #They are completely constrained given the location of the pixel and the fact
    #that the reflecting ray must be at a particular angle        
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    desired_light_direction_off_screen_towards_eye = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    return create_arc_helper(shell_point, screen_point, light_radius, arc_plane_normal, desired_light_direction_off_screen_towards_eye)
    
#just a continuation of the above function. allows you to pass in the normals so that this can work in taylor poly space
def create_arc_helper(shell_point, screen_point, light_radius, arc_plane_normal, desired_light_direction_off_screen_towards_eye):

    def f(point, t):
        point_to_screen_vec = _normalize(screen_point - point)
        surface_normal = _normalize(point_to_screen_vec + desired_light_direction_off_screen_towards_eye)
        derivative = _normalize(numpy.cross(surface_normal, arc_plane_normal))
        return derivative
    
    #TODO: this should really be based on light_radius...
    
    #estimate how long the piece of the shell will be (the one that is large enough to reflect all rays)
    #overestimates will waste time, underestimates cause it to crash :-P
    #note that we're doing this one half at a time
    def estimate_t_values():
        #TODO: make this faster if necessary by doing the following:
            #define the simple line that reflects the primary ray
            #intersect that with the max and min rays from the eye
            #check the distance between those intersections and double it or something
        t_step = 0.05
        if LOW_QUALITY_MODE:
            t_step = 0.5
        max_t = 5.0
        return numpy.arange(0.0, max_t, t_step)
    t_values = estimate_t_values()

    #use the vector field to define the exact shape of the surface (first half)
    half_arc = scipy.integrate.odeint(f, shell_point, t_values)
    
    #do the other half as well
    def g(point, t):
        return -1.0 * f(point, t)
    
    #combine them
    other_half_arc = list(scipy.integrate.odeint(g, shell_point, t_values))
    other_half_arc.pop(0)
    other_half_arc.reverse()
    return other_half_arc + list(half_arc)

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = numpy.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = numpy.linalg.lstsq(G, z)
    return m

#Note--might seem a little bizarre that we are transforming everything outside of PolyShell even though the details of its inner workings should be concealed
#but it's for efficiency reasons--matrix multiplying a bajillion points into the correct space is going to be way slower than just
#making them in the correct coordinate system in the first place
#really, should probably hide create_arc inside of PolyScale, but it's used elsewhere, so leaving it out for now
def make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec):
    """
    returns a non-trimmed scale patch based on the point (where the shell should be centered)
    angle_vec is passed in for our convenience, even though it is duplicate information (given the shell_point)
    """
    
    #taylor polys like to live in f(x,y) -> z
    #so build up the transformation so that the average of the shell -> screen vector and desired light vector is the z axis
    #eg, so the 0,0,0 surface normal is the z axis
    shell_to_screen_normal = _normalize(screen_point - shell_point)
    desired_light_dir = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    z_axis_world_dir = _normalize(desired_light_dir + shell_to_screen_normal)
    world_to_local_translation = -1.0 * shell_point
    world_to_local_rotation = numpy.zeros((3, 3))
    rotation_matrix.R_2vect(world_to_local_rotation, z_axis_world_dir, Point3D(0.0, 0.0, 1.0))
    #local_to_world_rotation = numpy.linalg.inv(world_to_local_rotation)
    local_to_world_rotation = numpy.zeros((3, 3))
    rotation_matrix.R_2vect(local_to_world_rotation, Point3D(0.0, 0.0, 1.0), z_axis_world_dir)
    
    def translate_to_local(p):
        return world_to_local_rotation.dot(p + world_to_local_translation)
    
    #convert everything into local coordinates
    transformed_light_dir = world_to_local_rotation.dot(desired_light_dir)
    h_arc_plane_normal = _get_arc_plane_normal(principal_ray, True)
    v_arc_plane_normal = _get_arc_plane_normal(principal_ray, False)
    transformed_screen_point = translate_to_local(screen_point)
    transformed_shell_point = Point3D(0.0, 0.0, 0.0)
    
    #actually go calculate the points that we want to use to fit our polynomial
    spine = create_arc_helper(transformed_shell_point, transformed_screen_point, light_radius, v_arc_plane_normal, transformed_light_dir)
    ribs = []
    for point in spine:
        rib = create_arc_helper(point, transformed_screen_point, light_radius, h_arc_plane_normal, transformed_light_dir)
        ribs.append(numpy.array(rib))
        
    points = numpy.vstack(ribs)
    #fit the polynomial to the points:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    coefficients = polyfit2d(x, y, z, order=10)
    order = int(numpy.sqrt(len(coefficients)))
    cohef = []
    for i in range(0, order):
        cohef.append(coefficients[i*order:(i+1)*order])
    cohef = numpy.array(cohef).copy(order='C')
    poly = hacks.taylor_poly.TaylorPoly(cohef=cohef.T)
    
    scale = PolyScale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        poly=poly,
        world_to_local_rotation=world_to_local_rotation,
        local_to_world_rotation=local_to_world_rotation,
        world_to_local_translation=world_to_local_translation,
        domain_cylinder_radius=light_radius
    )
    return scale

def make_old_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec):

    spine = create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=False)
    ribs = []
    for point in spine:
        rib = create_arc(principal_ray, point, screen_point, light_radius, angle_vec, is_horizontal=True)
        ribs.append(numpy.array(rib))
        
    #TODO: replace this with original:
    return Scale(
        shell_point=shell_point,
        pixel_point=screen_point,
        angle_vec=angle_vec,
        mesh=mesh.mesh_from_arcs(ribs)
    )


    #scale = mesh.Mesh(mesh.mesh_from_arcs(ribs))
    ##scale.export("temp.stl")
    #trimmed_scale = Scale(
    #    shell_point=shell_point,
    #    pixel_point=screen_point,
    #    angle_vec=angle_vec,
    #    mesh=mesh.trim_mesh_with_cone(scale._mesh, Point3D(0.0, 0.0, 0.0), _normalize(shell_point), light_radius)
    #)
    ##trimmed_scale.export("temp.stl")
    #return trimmed_scale
    
def calculate_error(scale, reference_scale):
    """
    I guess shoot rays all over the scale (from the pixel location), and see which also hit the reference scale, and get the distance
    least squares? or just sum all of it? I wonder why people use least squares all the time...
    note: will have to be average error per sample point, since different shells will have different number of sample points
    question is just whether to average the squares, or regularize them
    """
    start = reference_scale.pixel_point
    dist = 0.0
    num_hits = 0
    points = reference_scale.points()
    for point in points:
        end = 2.0 * (point - start) + start
        intersection_point, intersection_normal = scale.intersection_plus_normal(start, end)
        if intersection_point != None:
            num_hits += 1
            #print numpy.linalg.norm(intersection_point - point)
            dist += numpy.linalg.norm(intersection_point - point)
            #delta = intersection_point - point
            #dist += delta.dot(delta)
    average_error = dist / num_hits
    #print num_hits
    return average_error

def _get_scale_and_error_at_distance(distance, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec):
    shell_point = distance * angle_normal
    scale = make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec)
    error = max([calculate_error(scale, reference_scale) for reference_scale in reference_scales])
    scale.shell_distance_error = error
    return scale, error

#TODO: there are almost certainly better ways to optimize something like this, see numpy
def find_scale_and_error_at_best_distance(reference_scales, principal_ray, screen_point, light_radius, angle_vec):
    """
    iteratively find the best distance that this scale can be away from the reference scales
    """
    #seems pretty arbitrary, but honestly at that point the gains here are pretty marginal
    num_iterations = 14
    if LOW_QUALITY_MODE:
        num_iterations = 8
    angle_normal = angle_vector_to_vector(angle_vec, principal_ray)
    reference_distance = numpy.linalg.norm(reference_scales[0].shell_point)
    
    lower_bound_dist = reference_distance - light_radius
    upper_bound_dist = reference_distance + light_radius
    
    scales = {}
    def f(x):
        scale, error = _get_scale_and_error_at_distance(x, angle_normal, reference_scales, principal_ray, screen_point, light_radius, angle_vec)
        scales[x] = (scale, error)
        return error
    #best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=0.001, full_output=True, disp=0)
    best_value = scipy.optimize.fminbound(f, lower_bound_dist, upper_bound_dist, maxfun=num_iterations, xtol=0.001, full_output=False, disp=0)
    return scales[best_value]
    
def explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec):
    results = {}
    def f(x):
        pixel_point = prev_scale.pixel_point + x * optimization_normal
        scale, error = find_scale_and_error_at_best_distance([prev_scale], principal_ray, pixel_point, light_radius, angle_vec)
        results[x] = (scale, error)
        return error
    best_value, best_error, err, num_calls = scipy.optimize.fminbound(f, lower_bound, upper_bound, maxfun=num_iterations, xtol=0.0001, full_output=True, disp=3)
    return results[best_value]

def optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec):
    approximately_correct_scale, decent_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #print("Decent error: " + str(decent_error))
    
    #after that, simply find the point along that line (from the shell to that pixel) that is closest to the previous pixel
    #(since we don't want the screen to get any bigger than it has to)
    #and make the shell there
    #TODO: will have to look at how the optimization curves look for surfaces where we are optimizing against 3 surfaces...
    #might have to do another call to "explore_direction" to get the absolute best performance
    best_screen_point = closestPointOnLine(prev_scale.pixel_point, approximately_correct_scale.pixel_point, approximately_correct_scale.shell_point)
    #best_scale, error_for_best_scale = find_scale_and_error_at_best_distance([prev_scale], principal_ray, best_screen_point, light_radius, angle_vec)
    #print("'best' error: " + str(error_for_best_scale))
    
    #doing another crawl along the line because why not
    optimization_normal = _normalize(best_screen_point - prev_scale.pixel_point)
    final_scale, final_error = explore_direction(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #print("Final error: " + str(final_error))
    #scales.append(final_scale)
    return final_scale, final_error
    
def create_screen_mesh(ordered_scales):
    arc = [scale.pixel_point for scale in ordered_scales]
    left_arc = [p + Point3D(-1.0, 0.0, 0.0) for p in arc]
    right_arc = [p + Point3D(1.0, 0.0, 0.0) for p in arc]
    return mesh.Mesh(mesh=mesh.mesh_from_arcs([right_arc, arc, left_arc]))
    
def create_surface_via_scales(initial_shell_point, initial_screen_point, principal_ray):
    """
    Imagine a bunch of fish scales. Each represent a section of the shell, focused correctly for one pixel (eg, producing
    parallel rays heading towards the eye). By making a bunch of these, and adjusting the pixel locations so that they all line up,
    we should be able to make a surface that works well.
    
    Basic algorithm is a greedy one. Starting from the center scale, work outwards. Creates a hexagonal sort of mesh of these scales.
    Ask Josh for more details.
    """
    
    #based on the fact that your pupil is approximately this big
    #basically defines how big the region is that we are trying to put in focus with a given scale
    light_radius = 3.0
    fov = math.pi / 2.0
    #per whole the screen. So 90 steps for a 90 degree total FOV would be one step per degree
    total_phi_steps = 90
    
    #calculated:
    total_vertical_resolution = 2000
    min_pixel_spot_size = 0.005
    min_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * min_pixel_spot_size
    max_pixel_spot_size = 0.015
    max_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * max_pixel_spot_size
    
    #TODO: go delete the old scale stuff if this works!!!!!
    
    #create the first scale
    center_scale = make_old_scale(principal_ray, initial_shell_point, initial_screen_point, light_radius, AngleVector(0.0, 0.0))
    center_scale.shell_distance_error = 0.0
    #scales = [center_scale]
    scales = []
    
    new_center_scale = make_scale(principal_ray, initial_shell_point, initial_screen_point, light_radius, AngleVector(0.0, 0.0))
    new_center_scale.shell_distance_error = 0.0
    scales.append(new_center_scale)
    
    #create another scale right above it for debugging the error function
    #shell_point = initial_shell_point + Point3D(0.0, 3.0, -1.0)
    #angle_vec = AngleVector(math.pi/2.0, _normalized_vector_angle(principal_ray, _normalize(shell_point)))
    #other_scale = make_scale(principal_ray, shell_point, initial_screen_point+Point3D(0.0, -min_pixel_spot_size, min_pixel_spot_size), light_radius, angle_vec)
    #scales = [center_scale, other_scale]
    
    #calculate_error(other_scale, center_scale)
    
    #other_scale, error = find_scale_and_error_at_best_distance([center_scale], principal_ray,
    #    #initial_screen_point+Point3D(0.0, -10.0, 10.0), light_radius, angle_vec)
    #    initial_screen_point+Point3D(0.0, 0.0, 0.0), light_radius, angle_vec)
    #print error
    
    #wheeee
    #now let's make a grid of different pixel locations, and how those impact the final error
    #scales = [center_scale, other_scale]
    #scales = [center_scale]
    
    ##make a 5x5 grid, centered on the previous screen location, and with +/- reasonable spacing * 2 in either direction
    #spacing = (max_spacing + min_spacing) / 2.0
    #grid_size = 9
    #plot_x, plot_y = numpy.meshgrid(
    #    numpy.linspace(initial_screen_point[2] - 5.0, initial_screen_point[2], grid_size),
    #    numpy.linspace(initial_screen_point[1], initial_screen_point[1] + 5.0, grid_size))
    #error_values = numpy.zeros((grid_size,grid_size))
    #for i in range(0, grid_size):
    #    for j in range(0, grid_size):
    #        z = plot_x[i][j]
    #        y = plot_y[i][j]
    #        other_scale, error = find_scale_and_error_at_best_distance([center_scale], principal_ray,
    #            #TODO: has a 0 in there, which will not generalize
    #            Point3D(0.0, y, z), light_radius, angle_vec)
    #        print error
    #        error_values[i][j] = error
    #plot_error(plot_x, plot_y, error_values)
    
    ##ok, new approach to actually optimizing the next shell:
    ##simply walk along the direction orthogonal to the last pixel -> shell vector in the current plane
    ##and find the location with the minimal error
    #
    #lower_bound = 0.0
    ##NOTE: is a hack / guestimate
    #upper_bound = 2.0 * light_radius
    #num_iterations = 16
    #if LOW_QUALITY_MODE:
    #    num_iterations = 8
    #    
    #phi_step = 0.05
    #final_phi = fov/2.0
    #
    #for direction in (1.0, -1.0):
    #    phi = 0.0
    #    prev_scale = center_scale
    #    while phi < final_phi:
    #        phi += phi_step
    #        theta = math.pi / 2.0
    #        if direction < 0:
    #            theta = 3.0 * math.pi / 2.0
    #        angle_vec = AngleVector(theta, phi)
    #        #TODO: obviously this has to change in the general case
    #        optimization_normal = direction * numpy.cross(Point3D(1.0, 0.0, 0.0), _normalize(prev_scale.shell_point - prev_scale.pixel_point))
    #        scale, error = optimize_scale_for_angle(optimization_normal, lower_bound, upper_bound, num_iterations, prev_scale, principal_ray, light_radius, angle_vec)
    #        scales.append(scale)
    #        prev_scale = scale
    #        
    ##print out a little graph of the errors of the scales so we can get a sense
    ##NOTE: this shuffling is just so that the errors are printed in an intuitive order
    #num_scales = len(scales)
    #num_scales_in_arc = (num_scales - 1) / 2
    #lower_arc = scales[num_scales_in_arc+1:]
    #lower_arc.reverse()
    #ordered_scales = lower_arc + scales[:num_scales_in_arc+1]
    #print("theta  phi     error")
    #for scale in ordered_scales:
    #    print("%.2f %.2f    %.5f" % (scale.angle_vec.theta, scale.angle_vec.phi, scale.shell_distance_error))
    #    
    ##export all of the scales as one massive STL
    #merged_mesh = mesh.merge_meshes(ordered_scales)
    #mesh.Mesh(mesh=merged_mesh).export("all_scales.stl")
    #
    ##export the shape formed by the screen pixels as an STL
    #create_screen_mesh(ordered_scales).export("screen.stl")
    
    return scales

def plot_error(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_wireframe(x, y, z)
    plt.show()

def main():
    win = Window(23) # set the fps
    win.run()

if __name__ == '__main__':
    main()
    
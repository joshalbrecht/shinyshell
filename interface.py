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
from time import time, sleep

from OpenGL import GL, GLU
import pyglet
from pyglet.gl import *
import pyglet.window.key
import numpy
import scipy.integrate

import rotation_matrix
from optics import Point3D, _normalize, _normalized_vector_angle, _get_arc_plane_normal, angle_vector_to_vector, AngleVector, distToSegment
import mesh

#TODO: use the pyglet codes here instead
LEFT_MOUSE_BUTTON_CODE = 1L
MIDDLE_MOUSE_BUTTON_CODE = 2L
RIGHT_MOUSE_BUTTON_CODE = 4L

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
    
    def __init__(self, shell_pos, pixel_pos, num_rays, pupil_radius):
        self._shell = ReflectiveSurface(pos=shell_pos, color=(1.0, 1.0, 1.0), change_handler=self._on_change)
        self._pixel = ScreenPixel(pos=pixel_pos, color=(1.0, 1.0, 1.0), change_handler=self._on_change)
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
        self.sections.append(ShellSection(Point3D(0.0, 0.0, -60.0), Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
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
    def __init__(self, shell_pos=None, pixel_pos=None, angle_vec=None, **kwargs):
        mesh.Mesh.__init__(self, **kwargs)
        assert shell_pos != None
        assert pixel_pos != None
        assert angle_vec != None
        self._shell_pos = shell_pos
        self._pixel_pos = pixel_pos
        self._angle_vec = angle_vec
        
    @property
    def shell_pos(self):
        return self._shell_pos
    
    @property
    def pixel_pos(self):
        return self._pixel_pos
    
    @property
    def angle_vec(self):
        return self._angle_vec
    
    def points(self):
        point_data = self._mesh.GetPoints().GetData()
        return [numpy.array(point_data.GetTuple(i)) for i in range(0, point_data.GetSize())]

def create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=None):
    assert is_horizontal != None, "Must pass this parameter"
    
    #define a vector field for the surface normals of the shell.
    #They are completely constrained given the location of the pixel and the fact
    #that the reflecting ray must be at a particular angle        
    arc_plane_normal = _get_arc_plane_normal(principal_ray, is_horizontal)
    desired_light_direction_off_screen_towards_eye = -1.0 * angle_vector_to_vector(angle_vec, principal_ray)
    def f(point, t):
        point_to_screen_vec = _normalize(screen_point - point)
        surface_normal = _normalize((point_to_screen_vec + desired_light_direction_off_screen_towards_eye) / 2.0)
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
        t_step = 0.2
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
    
def make_scale(principal_ray, shell_point, screen_point, light_radius, angle_vec):
    """
    returns a non-trimmed scale patch based on the point (where the shell should be centered)
    angle_vec is passed in for our convenience, even though it is duplicate information (given the shell_point)
    """
    spine = create_arc(principal_ray, shell_point, screen_point, light_radius, angle_vec, is_horizontal=False)
    ribs = []
    for point in spine:
        rib = create_arc(principal_ray, point, screen_point, light_radius, angle_vec, is_horizontal=True)
        ribs.append(rib)
        
    scale = mesh.Mesh(mesh.mesh_from_arcs(ribs))
    #scale.export("temp.stl")
    trimmed_scale = Scale(
        shell_pos=shell_point,
        pixel_pos=screen_point,
        angle_vec=angle_vec,
        mesh=mesh.trim_mesh_with_cone(scale._mesh, Point3D(0.0, 0.0, 0.0), _normalize(shell_point), light_radius)
    )
    #trimmed_scale.export("temp.stl")
    return trimmed_scale
    
def calculate_error(scale, reference_scale):
    """
    I guess shoot rays all over the scale (from the pixel location), and see which also hit the reference scale, and get the distance
    least squares? or just sum all of it? I wonder why people use least squares all the time...
    note: will have to be average error per sample point, since different shells will have different number of sample points
    question is just whether to average the squares, or regularize them
    """
    start = reference_scale.pixel_pos
    dist = 0.0
    num_hits = 0
    points = reference_scale.points()
    for point in points:
        end = 2.0 * (point - start) + start
        intersection_point, intersection_normal = scale.intersection_plus_normal(start, end)
        if intersection_point != None:
            num_hits += 1
            dist += numpy.linalg.norm(intersection_point - point)
    average_error = dist / num_hits
    return average_error
    
def create_scale(phi, theta, prev_scale, principal_ray, dist_range, spacing_range):
    """
    actually creates a whole bunch of scales, evaluates each, and returns the best
    """
    #make a bunch of possible pixel locations (within the ranges)
    #for each pixel location:
        #find the shell distance that minimizes the error (subdivision search)
        #if that minimal error is the best so far, remember
    
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
    parallel_light_cylinder_radius = 3.0
    fov = math.pi / 2.0
    #per whole the screen. So 90 steps for a 90 degree total FOV would be one step per degree
    total_phi_steps = 90
    
    #calculated:
    total_vertical_resolution = 2000
    min_pixel_spot_size = 0.005
    min_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * min_pixel_spot_size
    max_pixel_spot_size = 0.015
    max_spacing = (float(total_vertical_resolution) / float(total_phi_steps)) * max_pixel_spot_size
    
    #create the first scale
    center_scale = make_scale(principal_ray, initial_shell_point, initial_screen_point, parallel_light_cylinder_radius, AngleVector(0.0, 0.0))
    
    #create another scale right above it for debugging the error function
    shell_point = initial_shell_point + Point3D(0.0, 3.0, -1.0)
    angle_vec = AngleVector(math.pi/2.0, _normalized_vector_angle(principal_ray, _normalize(shell_point)))
    other_scale = make_scale(principal_ray, shell_point, initial_screen_point+Point3D(0.0, -min_pixel_spot_size, min_pixel_spot_size), parallel_light_cylinder_radius, angle_vec)
    scales = [center_scale, other_scale]
    
    calculate_error(other_scale, center_scale)
    
    ##for now, we're just going to go up and down so we can visualize in 2D
    #prev_scale = center_scale
    #theta = math.pi / 2.0
    #assert num_phi_steps % 2 == 0, "Please just make it even so that it works nicely for both halves"
    #phi_values = numpy.linspace(0, fov / 2.0, num=num_phi_steps/2)[1:]
    #for phi in phi_values:
    #    #TODO: those distance ranges are completely arbitrary (they define curvature).
    #    #should really pull them out into a more sensible parameter
    #    scale = create_scale(phi, theta, prev_scale, principal_ray, dist_range=(-1.0, 2.0), spacing_range=(min_spacing, max_spacing))
    #    scales.append(scale)
    #    prev_scale = scale
    
    #TODO: create all of the rows instead of just going up
    ##create the row right, then left
    #grid = numpy.zeros(())
    #for i in range():
    #    scale = create_optimal_scale(phi, theta, scale_to_optimize_against, principal_ray, pixel_bounds)
    ##extend upward (dual create row because every odd and even row are different)
    ##extend downward
    
    return scales

def main():
    win = Window(23) # set the fps
    win.run()

if __name__ == '__main__':
    main()
    
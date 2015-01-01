#!/usr/bin/python

"""
Installation instructions: sudo pip install pyglet

This is a hacked together application to directly manipulate the surface in 2D.

All assumptions and coordinates are the same as in main.py. It's simply 2D because we're viewing a 3D scene from (zoom, 0, 0) looking at (0, 0, 0)
"""

import pyglet
from time import time, sleep

from main import Point3D, _normalize, _normalized_vector_angle, _get_arc_plane_normal, angle_vector_to_vector, AngleVector

from OpenGL import GL, GLU
from pyglet.gl import *

import pyglet.window.key

import numpy
import scipy.integrate

import math
import sys

LEFT_MOUSE_BUTTON_CODE = 1L
MIDDLE_MOUSE_BUTTON_CODE = 2L
RIGHT_MOUSE_BUTTON_CODE = 4L

def dist2(v, w):
    return sum(((math.pow(v[i] - w[i], 2) for i in range(0, len(v)))))

def distToSegmentSquared(p, v, w):
    l2 = dist2(v, w)
    if (l2 == 0):
        return dist2(p, v)
    n = w - v
    t = ((p - v).dot(n)) / l2
    if (t < 0):
        return dist2(p, v)
    if (t > 1):
        return dist2(p, w)
    return dist2(p, v + t * n)

def distToSegment(p, v, w):
    return math.sqrt(distToSegmentSquared(p, v, w))

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
        fixed_phi = _normalized_vector_angle(principal_ray, _normalize(self._shell.pos))
        
        #define a vector field for the surface normals of the shell.
        #They are completely constrained given the location of the pixel and the fact
        #that the reflecting ray must be at a particular angle        
        arc_plane_normal = _get_arc_plane_normal(principal_ray, False)
        def f(point, t):
            if point[1] > 0.0:
                theta = 3.0 * math.pi / 2.0
            else:
                theta = math.pi / 2.0
            eye_to_point_vec = angle_vector_to_vector(AngleVector(theta, fixed_phi), principal_ray)
            
            point_to_eye_vec = eye_to_point_vec * -1
            point_to_screen_vec = _normalize(self._pixel.pos - point)
            surface_normal = _normalize((point_to_screen_vec + point_to_eye_vec) / 2.0)
            derivative = _normalize(numpy.cross(surface_normal, arc_plane_normal))
            return derivative
        
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
        half_arc = scipy.integrate.odeint(f, self._shell.pos, t_values)
        
        #do the other half as well
        def g(point, t):
            return -1.0 * f(point, t)
        
        #combine them
        other_half_arc = list(scipy.integrate.odeint(g, self._shell.pos, t_values))
        other_half_arc.pop(0)
        other_half_arc.reverse()
        shell_points = other_half_arc + list(half_arc)
        
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
        self.zoom_distance = 100.0
        self.focal_point = Point3D(0,0,0)
        
        self.num_rays = 3
        self.pupil_radius = 2.0
        self.sections = []
        
        self.create_initial_sections()
        
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
            location = self._mouse_to_work_plane(x, y)
            self.sections.append(ShellSection(location, Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
            
        elif button == LEFT_MOUSE_BUTTON_CODE:
            self.left_click = x,y
        
            #pyglet.window.key.MOD_SHIFT
            ray = self._click_to_ray(x, y)
            obj = SceneObject.pick_object(ray)
            if obj:
                self.selection = [obj]
            else:
                self.selection = []
                
        elif button == MIDDLE_MOUSE_BUTTON_CODE:
            self.middle_click = x,y

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        start_plane_location = self._mouse_to_work_plane(x, y)
        end_plane_location = self._mouse_to_work_plane(x+dx, y+dy)
        delta = end_plane_location - start_plane_location
        
        if self.left_click:
            for obj in self.selection:
                obj.pos += delta
        if self.middle_click:
            self.focal_point += delta
                
    def _mouse_to_work_plane(self, x, y):
        ray = self._click_to_ray(x, y)
        point = Plane(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)).intersect_line(ray.start, ray.end)
        return point

    def on_mouse_release(self, x, y, button, modifiers):
        if button == LEFT_MOUSE_BUTTON_CODE:
            self.left_click = None
        elif button == MIDDLE_MOUSE_BUTTON_CODE:
            self.middle_click = None
        
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
        gluLookAt(self.zoom_distance, self.focal_point[1], self.focal_point[2],
                  0.0, self.focal_point[1], self.focal_point[2],
                  0, 1, 0)
        
        self._draw_axis()
        
        for section in self.sections:
            section.render()
        
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
        gluPerspective(65, width / float(height), 0.1, 1000)
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
            
def main():
    win = Window(23) # set the fps
    win.run()

if __name__ == '__main__':
    main()
    
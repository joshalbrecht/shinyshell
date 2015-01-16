
"""
Objects for the user to manipulate within the window
"""

import sys

import numpy
import OpenGL.GL

#this is the one thing that is allowed to import *
from optics.base import *
import optics.calculations

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
        OpenGL.GL.glColor3f(*self._color)
        
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
    
class RenderableArc(SceneObject):
    
    def __init__(self, arc, **kwargs):
        SceneObject.__init__(self, pos=arc.arc_plane.local_to_world(arc.start_point), **kwargs)
        self.arc = arc
        self.points = self.points()
        
    def points(self):
        return [self.arc.arc_plane.local_to_world(self.arc._local_to_plane(Point2D(x, self.arc._poly(x)))) for x in numpy.linspace(0.0, self.arc.max_x, 100)]
        
    def render(self):
        SceneObject.render(self)
        OpenGL.GL.glPointSize(5.0);
        OpenGL.GL.glBegin(OpenGL.GL.GL_POINTS)
        for point in self.points:
            OpenGL.GL.glVertex3f(*point)
        OpenGL.GL.glEnd()
    
class MovablePoint(SceneObject):
    def render(self):
        SceneObject.render(self)
        OpenGL.GL.glPointSize(5.0);
        OpenGL.GL.glBegin(OpenGL.GL.GL_POINTS)
        OpenGL.GL.glVertex3f(*self._pos)
        OpenGL.GL.glEnd()

class ScreenStartingPoint(MovablePoint): pass
class ScreenNormalPoint(MovablePoint): pass
class ShellStartingPoint(MovablePoint):
    @MovablePoint.pos.setter
    def pos(self, value):
        changed_value = Point3D(0.0, 0.0, value[2])
        self._pos = changed_value
        self.on_change()

class ScreenPixel(MovablePoint): pass

class ReflectiveSurface(SceneObject):
    """
    :attr segments: VisibleLineSegments connecting shell points
    """
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
        #point_to_eye = normalize(Point3D(0,0,0) - self._shell.pos)
        #point_to_pixel = normalize(self._pixel.pos - self._shell.pos)
        #surface_normal = normalize((point_to_eye + point_to_pixel) / 2.0)
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
        phi = normalized_vector_angle(principal_ray, normalize(self._shell.pos))
        if self._shell.pos[1] < 0.0:
            theta = 3.0 * math.pi / 2.0
        else:
            theta = math.pi / 2.0
        
        shell_points = optics.calculations.create_arc(principal_ray, self._shell.pos, self._pixel.pos, self.pupil_radius, AngleVector(theta, phi), is_horizontal=False)
        
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
                tangent = normalize(seg.end - seg.start)
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
        

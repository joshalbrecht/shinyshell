#!/usr/bin/python

"""
Installation instructions: sudo pip install pyglet

This is a hacked together application to directly manipulate the surface in 2D.

All assumptions and coordinates are the same as in main.py. It's simply 2D because we're viewing a 3D scene from (zoom, 0, 0) looking at (0, 0, 0)
"""

import pyglet
from time import time, sleep

from main import Point3D, _normalize

from OpenGL import GL, GLU
from pyglet.gl import *

class LineSegment(object):
    def __init__(self, start, end, color=(1.0, 1.0, 1.0)):
        self._start = start
        self._end = end
        self._color = color
        
    def render(self):
        glBegin(GL_LINES)
        glColor3f(*self._color)
        glVertex3f(*self._start)
        glVertex3f(*self._end)
        glEnd()

class Ray(LineSegment):
    def __init__(self, start, end):
        LineSegment.__init__(self, start, end, color=(0.5, 0.5, 0.5))

class SceneObject(object):
    """
    :attr pos: the position of the object in real coordinates (mm)
    :attr change_handler: will be called if the position changes
    """
    
    def __init__(self, pos=None, color=(1.0, 1.0, 1.0), change_handler=None):
        assert pos != None, "Must define a position for a SceneObject"
        self._pos = pos
        self._color = color
        self._change_handler = change_handler
        
    def on_change(self):
        if self._change_handler:
            self._change_handler()
            
    def render(self):
        glColor3f(*self._color)
        
    @property
    def pos(self): 
        return self._pos

    @pos.setter
    def pos(self, value): 
        self._pos = value
        self.on_change()

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
        self._shell = ReflectiveSurface(pos=shell_pos, color=(1.0, 1.0, 1.0), change_handler=self._recalculate)
        self._pixel = ScreenPixel(pos=pixel_pos, color=(1.0, 1.0, 1.0), change_handler=self._recalculate)
        self._num_rays = num_rays
        self._pupil_radius = pupil_radius
        
        #calculated attributes
        self._rays = []
        
        self._recalculate()
        
    def render(self):
        """Draw the shell section, pixel, and rays"""
        self._shell.render()
        self._pixel.render()
        for ray in self._rays:
            ray.render()
            
    def _recalculate(self):
        """Update internal state when any of the interesting variables have changed"""
        
        #note: this is just a hacky implementation right now to see if things are generally working
        point_to_eye = _normalize(Point3D(0,0,0) - self._shell.pos)
        point_to_pixel = _normalize(self._pixel.pos - self._shell.pos)
        surface_normal = _normalize((point_to_eye + point_to_pixel) / 2.0)
        tangent = Point3D(0.0, -1.0 * surface_normal[2], surface_normal[1])
        start = self._shell.pos + tangent
        end = self._shell.pos - tangent
        segment = LineSegment(start, end)
        self._shell.segments = [segment]
        self._rays = [Ray(Point3D(0,0,0), self._shell.pos), Ray(self._shell.pos, self._pixel.pos)]
        
        #TODO: do the right thing
        #figure out the angle of the primary ray
        #define a vector field for the surface normals of the shell. They are completely constrained given the location of the pixel and the fact that the reflecting ray must be at a particular angle
        #use that vector field to define the exact shape of the surface
        #simply pre-create a list of rays and line segments for the shell (all to be rendered later)

    @property
    def num_rays(self): 
        return self._num_rays

    @num_rays.setter
    def num_rays(self, value): 
        self._num_rays = value
        self._recalculate()
        
    @property
    def pupil_radius(self): 
        return self._pupil_radius

    @pupil_radius.setter
    def pupil_radius(self, value): 
        self._pupil_radius = value
        self._recalculate()

class Window(pyglet.window.Window):
    def __init__(self, refreshrate):
        super(Window, self).__init__(vsync = False)
        self.frames = 0
        self.framerate = pyglet.text.Label(text='Unknown', font_name='Verdana', font_size=8, x=10, y=10, color=(255,255,255,255))
        self.last = time()
        self.alive = 1
        self.refreshrate = refreshrate
        self.click = None
        self.drag = False
        
        self.zoom_distance = 100.0
        
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
        self.click = x,y

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.click:
            self.drag = True
            print 'Drag offset:',(dx,dy)

    def on_mouse_release(self, x, y, button, modifiers):
        if not self.drag and self.click:
            print 'You clicked here', self.click, 'Relese point:',(x,y)
        else:
            print 'You draged from', self.click, 'to:',(x,y)
        self.click = None
        self.drag = False
        
        print self._click_to_ray(x, y)
        
    def _click_to_ray(self, x, y):
        model = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        proj = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        view = GL.glGetIntegerv(GL.GL_VIEWPORT)
        start = GLU.gluUnProject(x, y, 0.0, model=model, proj=proj, view=view)
        end = GLU.gluUnProject(x, y, 1.0, model=model, proj=proj, view=view)
        return (start, end)

    def render(self):
        self.clear()
        
        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(self.zoom_distance, 0, 0, 0, 0, 0, 0, 1, 0)
        
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
    
#!/usr/bin/python
import pyglet
from time import time, sleep

from OpenGL import GL, GLU
from pyglet.gl import *

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
        gluLookAt(10, 0, 0, 0, 0, 0, 0, 1, 0)
        glBegin(GL_TRIANGLES)
        glVertex3f(0, 0, 0.0)
        glVertex3f(0, 0, -2.0)
        glVertex3f(0, 2.0, -2.0)
        glEnd()
        
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
            #  Without self.dispatc_events() the screen will freeze
            #  due to the fact that i don't call pyglet.app.run(),
            #  because i like to have the control when and what locks
            #  the application, since pyglet.app.run() is a locking call.
            event = self.dispatch_events()
            sleep(1.0/self.refreshrate)

win = Window(23) # set the fps
win.run()

import math
import sys
import time

import OpenGL.GL
import OpenGL.GLU
import numpy
import pyglet
import pyglet.window.key
import pyglet.window.mouse
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import PIL

#this is the one thing that is allowed to import *
from optics.base import *
import optics.rotation_matrix
import viewer.scene_objects

class Window(pyglet.window.Window):
    def __init__(self, refreshrate, generate_surface, stop_generating_surface):
        super(Window, self).__init__(vsync = False)
        self.frames = 0
        self.framerate = pyglet.text.Label(text='Unknown', font_name='Verdana', font_size=8, x=10, y=10, color=(255,255,255,255))
        self.last = time.time()
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
        
        self.scales = []
        
        self._generate_surface = generate_surface
        self._stop_generating_surface = stop_generating_surface
        
        #self._shell_point = Point3D(0.0, 0.0, -60.0)
        #self._screen_point = Point3D(0.0, 40.0, -20.0)
        self._principal_ray = Point3D(0.0, 0.0, -1.0)
        #self._screen_normal_point = self._screen_point + normalize(Point3D(0.0, -1.0, -1.0))
        
        self._shell_point = viewer.scene_objects.ShellStartingPoint(pos=Point3D(0.0, 0.0, -60.0), color=(1.0, 0.0, 0.0), change_handler=self.on_initial_parameter_change)
        self._screen_point = viewer.scene_objects.ScreenStartingPoint(pos=Point3D(0.0, 40.0, -20.0), color=(1.0, 0.0, 0.0), change_handler=self.on_initial_parameter_change)
        self._screen_normal_point = viewer.scene_objects.ScreenNormalPoint(pos=self._screen_point.pos + 10.0*normalize(Point3D(0.0, -1.0, -1.0)), color=(1.0, 0.0, 0.0), change_handler=self.on_initial_parameter_change)
        
        self.on_done_moving_things()
        
    def on_initial_parameter_change(self):
        pass
        
    def on_done_moving_things(self):
        def on_done(scales):
            self.scales = scales
        def on_new_scale(scale):
            self.scales.append(scale)
        self.scales = []
        self._stop_generating_surface()
        screen_normal = normalize(self._screen_normal_point.pos - self._screen_point.pos)
        self._generate_surface(self._shell_point.pos, self._screen_point.pos, screen_normal, self._principal_ray, on_done, on_new_scale)

    def on_draw(self):
        self.render()
        
    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.RIGHT:
            location = self._mouse_to_2d_plane(x, y)
            self.sections.append(viewer.scene_objects.ShellSection(location, Point3D(0.0, 40.0, -20.0), self.num_rays, self.pupil_radius))
            
        elif button == pyglet.window.mouse.LEFT:
            self.left_click = x,y
        
            ray = self._click_to_ray(x, y)
            obj = viewer.scene_objects.SceneObject.pick_object(ray)
            if obj:
                self.selection = [obj]
            else:
                self.selection = []
                
        elif button == pyglet.window.mouse.MIDDLE:
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
        view_normal = normalize(self.focal_point - self.camera_point)
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
        self.up_vector = normalize((matrix.dot(up_point) + self.focal_point) - self.camera_point)
        
    def _mouse_to_work_plane(self, x, y):
        ray = self._click_to_ray(x, y)
        point = Plane(self.focal_point, normalize(self.camera_point - self.focal_point)).intersect_line(ray.start, ray.end)
        return point
    
    def _mouse_to_2d_plane(self, x, y):
        ray = self._click_to_ray(x, y)
        point = Plane(Point3D(0.0, 0.0, 0.0), Point3D(1.0, 0.0, 0.0)).intersect_line(ray.start, ray.end)
        return point

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.left_click = None
            self.on_done_moving_things()
        elif button == pyglet.window.mouse.MIDDLE:
            self.middle_click = None
            
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        distance = numpy.linalg.norm(self.camera_point - self.focal_point)
        for i in range(0, abs(scroll_y)):
            if scroll_y < 0:
                distance *= self.zoom_multiplier
            else:
                distance *= 1.0 / self.zoom_multiplier
        self.camera_point = self.focal_point + distance * normalize(self.camera_point - self.focal_point)
        
    def _click_to_ray(self, x, y):
        model = OpenGL.GL.glGetDoublev(OpenGL.GL.GL_MODELVIEW_MATRIX)
        proj = OpenGL.GL.glGetDoublev(OpenGL.GL.GL_PROJECTION_MATRIX)
        view = OpenGL.GL.glGetIntegerv(OpenGL.GL.GL_VIEWPORT)
        start = Point3D(*OpenGL.GLU.gluUnProject(x, y, 0.0, model=model, proj=proj, view=view))
        end = Point3D(*OpenGL.GLU.gluUnProject(x, y, 1.0, model=model, proj=proj, view=view))
        return Ray(start, end)

    def _draw_axis(self):
        axis_len = 10.0
        
        OpenGL.GL.glBegin(OpenGL.GL.GL_LINES)
        
        OpenGL.GL.glColor3f(1.0, 0.0, 0.0)
        OpenGL.GL.glVertex3f(0.0, 0.0, 0.0)
        OpenGL.GL.glVertex3f(axis_len, 0.0, 0.0)
        
        OpenGL.GL.glColor3f(0.0, 1.0, 0.0)
        OpenGL.GL.glVertex3f(0.0, 0.0, 0.0)
        OpenGL.GL.glVertex3f(0.0, axis_len, 0.0)
        
        OpenGL.GL.glColor3f(0.0, 0.0, 1.0)
        OpenGL.GL.glVertex3f(0.0, 0.0, 0.0)
        OpenGL.GL.glVertex3f(0.0, 0.0, axis_len)
        
        OpenGL.GL.glEnd()

    def _draw_face(self):
        img = PIL.Image.open('./head_image.png')
        img_data = numpy.array(list(img.getdata()), numpy.int8)
        texture_id = OpenGL.GL.glGenTextures(1)
        OpenGL.GL.glPixelStorei(OpenGL.GL.GL_UNPACK_ALIGNMENT, 1)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, texture_id)        
        OpenGL.GL.glTexParameterf(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_S, OpenGL.GL.GL_CLAMP)
        OpenGL.GL.glTexParameterf(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_WRAP_T, OpenGL.GL.GL_CLAMP)
        OpenGL.GL.glTexParameterf(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MAG_FILTER, OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glTexParameterf(OpenGL.GL.GL_TEXTURE_2D, OpenGL.GL.GL_TEXTURE_MIN_FILTER, OpenGL.GL.GL_LINEAR)
        OpenGL.GL.glTexImage2D(OpenGL.GL.GL_TEXTURE_2D, 0, OpenGL.GL.GL_RGB, img.size[0], img.size[1], 0,
                               OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE, img_data)

        OpenGL.GL.glEnable(OpenGL.GL.GL_TEXTURE_2D)
        OpenGL.GL.glBindTexture(OpenGL.GL.GL_TEXTURE_2D, texture_id)
        
        OpenGL.GL.glPushMatrix()
        OpenGL.GL.glTranslatef(0.0, 100.0, 0.0)
        
        OpenGL.GL.glBegin(OpenGL.GL.GL_TRIANGLES)

        OpenGL.GL.glColor3f(1.0, 1.0, 1.0)

        # 4px on image = 1mm
        img_height = float(img.size[1])
        img_width = 900.0
        coord_height = img_height/4
        coord_width = img_width/4

        # image is drawn where 0,0 is top left corner
        # make first triangle, top left half
        OpenGL.GL.glTexCoord2f(0, 0)
        OpenGL.GL.glVertex3f(0, 0, coord_width)
        OpenGL.GL.glTexCoord2f(0, 1)
        OpenGL.GL.glVertex3f(0, -coord_height, coord_width)        
        OpenGL.GL.glTexCoord2f(img_width/img.size[0], 1)
        OpenGL.GL.glVertex3f(0, -coord_height, 0)

        # make second triangle, bottom right half
        OpenGL.GL.glTexCoord2f(img_width/img.size[0], 0)
        OpenGL.GL.glVertex3f(0, 0, 0)
        OpenGL.GL.glTexCoord2f(0, 0)
        OpenGL.GL.glVertex3f(0, 0, coord_width)
        OpenGL.GL.glTexCoord2f(img_width/img.size[0], 1)
        OpenGL.GL.glVertex3f(0, -coord_height, 0)
        
        OpenGL.GL.glEnd()
        OpenGL.GL.glPopMatrix()

    def render(self):
        
        self.clear()
        
        OpenGL.GL.glClear(OpenGL.GL.GL_COLOR_BUFFER_BIT)
        OpenGL.GL.glLoadIdentity()
        OpenGL.GLU.gluLookAt(self.camera_point[0], self.camera_point[1], self.camera_point[2],
                  self.focal_point[0], self.focal_point[1], self.focal_point[2],
                  self.up_vector[0], self.up_vector[1], self.up_vector[2])
        
        self._draw_axis()
        self._draw_face()
        
        for section in self.sections:
            section.render()
            
        for scale in self.scales:
            scale.render()
            
        self._shell_point.render()
        self._screen_point.render()
        self._screen_normal_point.render()
        
        if time.time() - self.last >= 1:
            self.framerate.text = str(self.frames)
            self.frames = 0
            self.last = time.time()
        else:
            self.frames += 1
        self.framerate.draw()
        self.flip()
        
    def on_resize(self, width, height):
        OpenGL.GL.glViewport(0, 0, width, height)
        OpenGL.GL.glMatrixMode(OpenGL.GL.GL_PROJECTION)
        OpenGL.GL.glLoadIdentity()
        OpenGL.GLU.gluPerspective(65, width / float(height), 0.01, 500)
        OpenGL.GL.glMatrixMode(OpenGL.GL.GL_MODELVIEW)
        return pyglet.event.EVENT_HANDLED

    def on_close(self):        
        self.alive = 0
        self._stop_generating_surface()
        import os
        os._exit(0)

    def run(self):
        while self.alive:
            self.render()
            # ----> Note: <----
            #  Without self.dispatch_events() the screen will freeze
            #  due to the fact that i don't call pyglet.app.run(),
            #  because i like to have the control when and what locks
            #  the application, since pyglet.app.run() is a locking call.
            event = self.dispatch_events()
            time.sleep(1.0/self.refreshrate)
            

#!/usr/bin/python

"""
Installation instructions: sudo pip install pyglet
(actually make sure you install everything from main.py as well)

This application is meant to be a complete replacement for main.py (except I'm trying to never use pyoptools)
All assumptions and coordinates are the same as in main.py.

Usage instructions:

left click and drag to move screen pixels or shell sections
right click to make a new piece of shell (and paired pixel)
middle mouse drag to pan
middle mouse roll to zoom
shift + middle mouse to rotate the world
"""

#by installing this, we can automatically import .pyx files without having to set up some crazy build step.
#nice and convenient
import pyximport
pyximport.install()

#this is the one thing that is allowed to import *
from optics.base import *
import optics.calculations
import viewer.window

def main():
    frames_per_second = 23
    win = viewer.window.Window(frames_per_second)
    initial_shell_point = Point3D(0.0, 0.0, -60.0)
    initial_screen_point = Point3D(0.0, 40.0, -20.0)
    principal_ray = Point3D(0.0, 0.0, -1.0)
    win.scales = optics.calculations.create_surface_via_scales(initial_shell_point, initial_screen_point, principal_ray)
    win.run()

if __name__ == '__main__':
    main()
    
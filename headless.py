#!/usr/bin/python

"""
runs without the gui
"""

import multiprocessing
import threading

import pyximport

#by installing this, we can automatically import .pyx files without having to set up some crazy build step.
#nice and convenient
pyximport.install()

#this is the one thing that is allowed to import *
from optics.base import *
import optics.calculations

NUM_PROCESSES = 4

def main():
    process_pool = multiprocessing.Pool(NUM_PROCESSES)
    stop_flag = threading.Event()
    def on_new_scale(scale):
        pass
    principal_ray = Point3D(0.0, 0.0, -1.0)
    shell_point = Point3D(0.0, 0.0000000000, -82.7510541100)
    screen_point = Point3D(0.0, 74.6275528483, -63.2394043192)
    screen_normal = Point3D(0., -0.9850546720, 0.1722419612)
    scales = optics.calculations.create_surface_via_scales(shell_point, screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_scale)

if __name__ == '__main__':
    main()

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

import threading
import multiprocessing

import pyximport

#by installing this, we can automatically import .pyx files without having to set up some crazy build step.
#nice and convenient
pyximport.install()

#this is the one thing that is allowed to import *
from optics.base import *
import optics.calculations
import viewer.window

NUM_PROCESSES = 4

master_thread = None
process_pool = None
stop_flag = threading.Event()

def clear_temporary_data():
    """
    Use this space to delete everything in a temporary folder, tec
    """
    pass

def generate_surface(shell_point, screen_point, screen_normal, principal_ray, on_done, on_new_scale):
    global master_thread, process_pool, stop_flag
    
    assert master_thread == None
    assert process_pool == None
    stop_flag.clear()
    
    clear_temporary_data()
    
    process_pool = multiprocessing.Pool(NUM_PROCESSES)
    
    def calculate():
        scales = optics.calculations.create_surface_via_scales(shell_point, screen_point, screen_normal, principal_ray, process_pool, stop_flag, on_new_scale)
        on_done(scales)
    
    master_thread = threading.Thread(target=calculate)
    master_thread.start()
    
def stop_generating_surface():
    global master_thread, process_pool, stop_flag
    
    if master_thread != None:
        stop_flag.set()

    if process_pool != None:
        process_pool.terminate()
        process_pool.join()
        process_pool = None
        
    if master_thread != None:
        master_thread.join()
        master_thread = None

def main():
    frames_per_second = 23
    win = viewer.window.Window(frames_per_second, generate_surface, stop_generating_surface)
    win.run()

if __name__ == '__main__':
    main()
    
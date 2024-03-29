
"""
Just a bunch of constants that define the behavior of the script.
"""

#and enable this for super ridiculously low quality
ULTRA_LOW_QUALITY_MODE = 0
#enable this to speed up development. Just cuts back on a lot of precision
LOW_QUALITY_MODE = 1
#this should be sort of the default for the actual surface
HIGH_QUALITY_MODE = 2

#QUALITY_MODE = ULTRA_LOW_QUALITY_MODE
#QUALITY_MODE = LOW_QUALITY_MODE
QUALITY_MODE = HIGH_QUALITY_MODE

#TODO: we should theoretically be able to use much higher orders. however, it ends up doing totally insane things.
#we have to think more about how we are fitting the fuction so that it doesn't explode in such a ridiculous way

#what order for taylor poly approximations. higher is better quality but slower
#POLY_ORDER = 3
#if QUALITY_MODE <= LOW_QUALITY_MODE:
#    POLY_ORDER = 2

POLY_ORDER = 3

#based on the fact that your pupil is approximately this big
#basically defines how big the region is that we are trying to put in focus with a given patch
LIGHT_RADIUS = 3.0

#defines how many ribs in each arc, roughly
NUM_SLICES = 10

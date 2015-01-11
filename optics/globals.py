
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

#what order for taylor poly approximations. higher is better quality but slower
POLY_ORDER = 3
if QUALITY_MODE <= LOW_QUALITY_MODE:
    POLY_ORDER = 2

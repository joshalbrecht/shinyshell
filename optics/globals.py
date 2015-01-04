
"""
Just a bunch of constants that define the behavior of the script.
"""

#no longer needed
#enable this to speed up development. Just cuts back on a lot of precision
LOW_QUALITY_MODE = True

#what order for taylor poly approximations. higher is better quality but slower
POLY_ORDER = 3
if LOW_QUALITY_MODE:
    POLY_ORDER = 2


"""
Mostly contains a bunch of flags about what you would like to debug.
"""

#whether to debug fitting 2D polynomials during Arc creation
POLYARC_FITTING = False

#whether to debug the intersections with PolyArcs
POLYARC_INTERSECTIONS = False

#whether to debug the creation of Arcs
ARC_CREATION = False

#whether to plot each rib as it is created
INDIVIDUAL_RIB_CREATION = False

#whether to debug the creation of the ribs that become the grid
RIB_CREATION = False

#whether to debug the taylor polynomial creation process
TAYLOR_SURFACE_CREATION = False

#whether to debug ray intersections with taylor surfaces
TAYLOR_SURFACE_REFLECTIONS = False

#whether to debug the focal error calculation for Taylor poly surfaces
TAYLOR_SURFACE_FOCAL_ERROR = False

#whether to debug the reflections for determining the next screen point from a patch
PATCH_FOCAL_REFLECTIONS = True

#whether to debug the focal size of the reflected rays when determining the next screen point from a patch
PATCH_FOCAL_SPOT_SIZE = True

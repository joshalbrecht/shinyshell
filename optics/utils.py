
"""
Code that helps write code goes here. Profilers, logging, etc
"""

import cProfile
import pstats

def profile_line(line, glb, lcl):
    filename= 'optics.profile'
    cProfile.runctx(line, glb, lcl, filename)    
    p = pstats.Stats(filename).strip_dirs()
    p.sort_stats('time').print_stats(30)
    p.sort_stats('cumulative').print_stats(30)
    
    import sys
    sys.exit()
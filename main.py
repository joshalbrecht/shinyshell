#!/user/bin/python

import math

import numpy

import wxversion
wxversion.select('2.8')
import wx

from matplotlib import use
use('WXAgg')

from pyoptools.all import *

def main():
    app = wx.PySimpleApp()
    shape=Circular(radius=20.)
    cohef = numpy.array([[0,0.01],[0.01,0.02]]).copy(order='C')
    S1=TaylorPoly(shape=shape, cohef=cohef)
    S2=Spherical(curvature=1/200., shape=Circular(radius=20.))
    S3=Cylinder(radius=20,length=6.997)
    L1=Component(surflist=[(S1, (0, 0, -5), (0, 0, 0)), (S2, (0, 0, 5), (0, math.pi, 0)), (S3,(0,0,.509),(0,0,0))], material=schott["BK7"])
    surflist=[(S1, (0, 0, -5), (0, 0, 0)),
              (S2, (0, 0, 5), (0, math.pi, 0)),
              (S3,(0,0,.509),(0,0,0))]
    ccd=CCD(size=(100,100), transparent=False)
    S=System(complist=[(L1, (0, 0, 20), (0, 0, 0)),(ccd, (0, 0, 150), (0, 0, 0))], n=1)
    R=[Ray(pos=(0, 0, 0), dir=(0, 0, 1)), Ray(pos=(10, 0, 0), dir=(0, 0, 1)), Ray(pos=(-10, 0, 0), dir=(0, 0, 1)),Ray(pos=(0, 10, 0), dir=(0, 0, 1)), Ray(pos=(0, -10, 0), dir=(0, 0, 1)),]
    S.ray_add(R)
    S.propagate()
    glPlotFrame(S)
    spot_diagram(ccd)
    app.MainLoop()

if __name__ == '__main__':
    main()

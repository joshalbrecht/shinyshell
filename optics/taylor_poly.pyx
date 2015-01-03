 #!/usr/bin/env python
# -*- coding: UTF-8 -*-
# cython: profile=True
#------------------------------------------------------------------------------
#Note: Josh hacked this file a bit so that it would actually work with later versions of cython, etc
#Note2: Josh modified this so that it doesn't even use (or work with) pyoptools anymore.
# Copyright (c) 2007, Ricardo Amezquita Orozco <AUTHOR>
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.
#
#
# Author:         Ricardo Amézquita Orozco
# Description:    Polynomical surface definition module
#------------------------------------------------------------------------------
#
'''Module that defines a polynomical optical surface, using taylos polynomials
   Note: The polynomials used here are diferent to the polynomials defined in the
   module misc.Poly2D. This has to be fixed.
'''
from numpy import  array, asarray, arange, polyadd, polymul, polysub, polyval,\
     dot, inf, roots, zeros, meshgrid, sqrt,where, abs,  isreal
     
from optics.base import *

cimport numpy as np
cimport cython


def polypow(p,n):
    """
    Function that calculates the coheficients of the n'th power of a 1d polynomials
    """
    if n==0: return [1.]
    elif n==1:return p
    else: return polymul(p,polypow(p,n-1))

def Poly_DyDx(Poly):
    """
    Function that calculates the derivative with respect to ``X`` and ``Y`` of a
    given polynomial represented as a matrix.

    Return Value


        (dx,dy)
            Tuple containing the matrix representing de derivative with
            respect to ``X``, and the matrix containing the derivative with
            respect to ``Y``
    """
    Dx=zeros((Poly.shape[0],Poly.shape[1]))
    Dy=zeros((Poly.shape[0],Poly.shape[1]))
    Pows=Poly.nonzero()
    y_pow=Pows[0]
    x_pow=Pows[1]
    for i in range(y_pow.shape[0]):
        if x_pow[i]!=0 or y_pow[i]!=0:
            Dy[y_pow[i]-1,x_pow[i]]=Poly[y_pow[i],x_pow[i]]*y_pow[i]
            Dx[y_pow[i],x_pow[i]-1]=Poly[y_pow[i],x_pow[i]]*x_pow[i]

    return Dx,Dy

def polymul2d(p1, p2):
    """Function to calculate the product of 2 2d polynomials

    The polynomials are represented by the 2 2d arrays p1, and p2

    The result is an array representing the resulting polynomial.
    """

    np1x, np1y=p1.shape
    np2x, np2y=p2.shape

    result=zeros((np1x+np2x-1, np1y+np2y-1))
    for i1 in range(np1x):
        for j1 in range(np1y):
            for i2 in range(np2x):
                for j2 in range(np2y):
                    result[i1+i2, j1+j2]=result[i1+i2, j1+j2]+p1[i1, j1]*p2[i2, j2]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def eval_poly(p, x,y):
    """Function that returns the value Z=poly(x,y), where poly is a matriz
    representing a polinomial
    """
    cdef int i

    x=asarray(x)
    y=asarray(y)
    Result=zeros(x.shape)
    Pows=array(p.nonzero()).copy(order='C')

    cdef np.ndarray[np.int_t, ndim=1, mode="c"]  y_pow=Pows[0]
    cdef np.ndarray[np.int_t, ndim=1, mode="c"]  x_pow=Pows[1]

    for i in range(y_pow.shape[0]):
        Result=Result+p[y_pow[i],x_pow[i]]*x**x_pow[i]*y**y_pow[i]
    return Result

cdef class TaylorPoly(object):
    """Class to define surfaces described by a Taylor polynomial
    
    TODO: There are some bad things that happen with very high order polys. Like crazy intersections out in the middle of nowhere...

    Description

    The TaylorPoly class is used to define surfaces that can be described using
    a Taylor polynomial.

    To define an TaylorPoly surface you should pass the shape of the apperture,
    and a coefficient matrix.

    The coheficient matrix is a numpy matrix, that holds information about the
    coheficients of the polynomial.

    cohef= [[x0y0, x1y0, x2y0,...,xmy0],

            [x0y1, x1y1, x2y1,...,xmy1],

            [x0y2, x1y2, x2y2,...,xmy2],

            [ ... , ... , ... ,..., ... ],

            [x0yn, x1yn, x2yn,...,xmyn]]

    using this coheficients, the polynomial is defined as::

        p(x,y)= x0y0+x1y0*X + ... + x0y1*Y +x1y1*X*Y+...+ xmyn*X^m*Y^n

    Example:

        >>> cs=TaylorPoly(shape=Rectangle(size=(5,5)), \
                        cohef =[[0,1],[1,2]])

    """
    #cohef=Array('d')
    #  [[ x0y0, x1y0, x2y0,.....],
    #  [[ x0y1, x1y1, x2y1,.....],
    #  [[ x0y2, x1y2, x2y2,.....],
    #  [[ x0y3, x1y3, x2y3,.....],
    #  [[ ... , ... , ... ,.....],
    #  [[ x0y.., x1y..,x2y., ...],
    cdef object cohef
    cdef object domain_radius
    cdef object domain_point
    cdef object domain_sq_radius

    def __init__(self,cohef=(0,0,0), domain_radius=None, domain_point=None):
        self.cohef = cohef
        #self.cohef=array(cohef)
        #self.addkey("cohef")
        self.domain_radius = domain_radius
        self.domain_point = normalize(domain_point)
        self.domain_sq_radius = domain_radius * domain_radius
        
    cpdef in_domain(self, p):
        n = self.domain_point
        delta = p - (p.dot(n) * n)
        return delta.dot(delta) < self.domain_sq_radius

    #~ def __reduce__(self):
        #~
        #~ args=(self.cohef, self.reflectivity, self.shape)
        #~ return(type(self),args,self.__getstate__())

    cpdef eval_poly(self, x,y):
        x=asarray(x)
        y=asarray(y)
        Result=zeros(x.shape)
        Pows=self.cohef.nonzero()
        y_pow=Pows[0]
        x_pow=Pows[1]
        for i in range(y_pow.shape[0]):
            Result=Result+self.cohef[y_pow[i],x_pow[i]]*x**x_pow[i]*y**y_pow[i]
        return Result

    cpdef topo(self, x, y):
        """**Returns the Z value for a given X and Y**

        This method returns the topography of the polynomical surface to be
        used to plot the surface.
        """
        return self.eval_poly(x, y)

    cpdef _intersection(self, start, direction):
        '''
        Point of intersection between a ray and the polynomial surface

        This method returns the point of intersection  between the surface
        and the ray. This intersection point is calculated in the coordinate
        system of the surface.

        start and direction must be in the coordinate system of the surface
        '''
        

        ## Polynomial parametric equations describing the beam
        ## x=m_x t +b_x
        ## y=m_y t +b_y
        ## z=m_z t +b_z
        ox,oy,oz=start
        dx,dy,dz=direction

        RX=[dx,ox]
        RY=[dy,oy]
        RZ=[dz,oz]

        ## generate solving polynomial ( replacing x , y, z with corresponding dependent expressions t)

        Pows=self.cohef.nonzero()
        y_pow=Pows[0]
        x_pow=Pows[1]
        ##replace X and Y in the equation of the surface
        result=[0] #initialize polynomial
        
        for i in range(y_pow.shape[0]):
            ## self.cohef[y_pow[i],x_pow[i]],"y^",y_pow[i],"x^",x_pow[i]
            tx=polypow(RX,x_pow[i])
            ty=polypow(RY,y_pow[i])
            tm=self.cohef[y_pow[i],x_pow[i]]*polymul(tx,ty)
            result=polyadd(result,tm)

        ##Z in place so that it is the expression f ( x (t ) , y (t) , z (t ) ) = 0
        result=polysub(result,RZ)

        # There is a problem when the coefficients are << 1 but not 0
        # Truncate the values to solve this problem
        #TODO: find another solution to this problem

        result=where(abs(result)<1e-14,0,result)

        ##Find the solutions of the expression
        r=roots(result)
        
        dist=inf
        ret_val=None#array((inf,inf,inf))
        for i in r:
            #Just consider positive solutions i
            #if i is negative , the surface is behind the lightning ...
            #no cutting , plus i must be real
            if i>=0. and isreal(i):
                i=i.real
                pc=array((polyval(RX,i),polyval(RY,i),polyval(RZ,i)))
                if self.in_domain(pc):
                    ## ray origin distance to the cutting point ( squared)
                    dist_c=dot(pc-start,pc-start)
                    #Find the minimum distance
                    if dist_c<dist:
                        ret_val=pc
                        dist=dist_c

        return ret_val


    cpdef np.ndarray normal(self,int_p):
        """**Return the vector normal to the surface**

        This method returns the vector normal to the polynomical surface at a
        point ``int_p=(x,y,z)``.

        Note: It uses ``x`` and ``y`` to calculate the ``z`` value and the normal.
        """
        xo,yo,zo=int_p

        Dx,Dy=Poly_DyDx(self.cohef)

        dx=eval_poly(Dx,xo,yo)
        dy=eval_poly(Dy,xo,yo)

        n=array((dx,dy,-1.))## Ojo, tambien puede ser -dx,-dy,1

        n=n/sqrt(dot(n,n))

        return n


    def _repr_(self):
        '''Return an string with the representation of the taylor polynomical surface
        '''
        return "TaylorPoly(cohef="+str(self.cohef)+")"

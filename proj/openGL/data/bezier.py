##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from numpy import ndarray, array, vander, dot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def uvander(self, u): 
    return vander(array(u, copy=False, ndmin=1), len(self))

class Bezier(ndarray):
    __array_priority__ = -1

    groups = {}
    order = {}

    uvander = uvander

    def at(self, pts, u):
        if isinstance(pts, basestring):
            pts = self.groups[pts]
        return dot(self.uvander(u), dot(self, pts))

    def atP(self, pts):
        if isinstance(pts, basestring):
            pts = self.groups[pts]
        Bp = dot(self, pts)
        Bp = Bp.view(BezierP)
        return Bp

    def atU(self, u):
        uB = dot(self.uvander(u), self)
        uB = uB.view(UBezier)
        uB.groups = self.groups
        return uB

class UBezier(ndarray):
    __array_priority__ = -1

    def atP(self, pts):
        if isinstance(pts, basestring):
            pts = self.groups[pts]
        return dot(self, pts)
    at = atP

class BezierP(ndarray):
    __array_priority__ = -1

    uvander = uvander
    def atU(self, u):
        return dot(self.uvander(u), self)
    at = atU

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for Mb in [
        array([[-1,  1], [ 1,  0]], 'b'),
        array([[ 1, -2,  1], [-2,  2,  0], [ 1,  0,  0]], 'b'),
        array([[-1,  3, -3,  1], [ 3, -6,  3,  0], [-3,  3,  0,  0], [ 1,  0,  0,  0]], 'b'),
        ]:
    Bezier.order[len(Mb)-1] = Mb.view(Bezier)

b1 = bLinear = Bezier.linear = Bezier.order[1]
b2 = bQuadratic = Bezier.quadratic = Bezier.order[2]
b3 = bCubic = Bezier.cubic = Bezier.order[3]

Bezier.cubic.groups = {
    'a0': array([0., 1., 0., 1.]),
    '~a0': array([1., 0., 1., 0.]),

    'a': array([0., 0.5, 0.5, 1.]),
    '~a': array([1., 0.5, 0.5, 0.]),
    'a1': array([0., 0.5, 0.5, 1.]),
    '~a1': array([1., 0.5, 0.5, 0.]),

    'a2': array([0., 0.25, 0.75, 1.]),
    '~a2': array([1., 0.75, 0.25, 0.]),

    'a3': array([0., 0.125, 0.875, 1.]),
    '~a3': array([1., 0.875, 0.125, 0.]),

    'a4': array([0., 0.0625, 0.9375, 1.]),
    '~a4': array([1., 0.9375, 0.0625, 0.]),

    'ak': array([0., 0., 1., 1.]),
    '~ak': array([1., 1., 0., 0.]),

    'fast': array([0., 1., 1., 1.]),
    '~fast': array([1., 0., 0., 0.]),

    'slow': array([0., 0., 0., 1.]),
    '~slow': array([1., 1., 1., 0.]),
    }


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import operator
import numpy
from xform import Transform3dh
from xform2d import Transform2dh
from vector import linearMapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Rect(object):
    def __init__(self, p0=None, p1=None, size=None):
        if p0:
            self.p0 = p0
        if p1:
            self.p1 = p1
        if size:
            self.size = size

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public properties
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getX0(self): return self.p0[0]
    def setX0(self, value): self.p0[0] = value
    x0 = property(getX0, setX0)

    def getY0(self): return self.p0[1]
    def setY0(self, value): self.p0[1] = value
    y0 = property(getY0, setY0)


    def getX1(self): return self.p1[0]
    def setX1(self, value): self.p1[0] = value
    x1 = property(getX1, setX1)

    def getY1(self): return self.p1[1]
    def setY1(self, value): self.p1[1] = value
    y1 = property(getY1, setY1)

    def getWidth(self): return self.p1[0] - self.p0[0]
    def setWidth(self, width): self.p1[0] = self.p0[0] + width
    width = property(getWidth, setWidth)

    def getHeight(self): return self.p1[0] - self.p0[0]
    def setHeight(self, height): self.p1[1] = self.p0[1] + height
    height = property(getHeight, setHeight)

    def getSize(self): return (self.getWidth(), self.getHeight())
    def setSize(self, size): 
        self.setWidth(size[0])
        self.getHeight(size[1])

    def getAspect(self): 
        return float(self.getHeight()) / float(self.getWidth())
    aspect = property(getAspect)

    def getBox(self):
        return self.p0, self.p1
    def setBox(self, box):
        self.p0, self.p1 = box

    def getXV(self):
        return self.p0[0], self.p1[0]
    xv = property(getXV)
    def getYV(self):
        return self.p0[1], self.p1[1]
    yv = property(getYV)

    def mapPointTo(self, point, *args, **kw):
        return self.mapPointsTo([point], *args, **kw)[0]

    def mapPointsTo(self, points, xspan=(-1., 1.), yspan=None, flipx=False, flipy=False):
        xspan, yspan = xspan or yspan, yspan or xspan
        if flipx: xspan = xspan[1], xspan[0]
        if flipy: yspan = yspan[1], yspan[0]

        xfn = linearMapping(self.xv, xspan)
        yfn = linearMapping(self.yv, yspan)
        return [(xfn(x),yfn(y)) for x,y in points]

    def mapPointFrom(self, point, *args, **kw):
        return self.mapPointsFrom([point], *args, **kw)[0]

    def mapPointsFrom(self, points, xspan=(-1., 1.), yspan=None, flipx=False, flipy=False):
        xspan, yspan = xspan or yspan, yspan or xspan
        if flipx: xspan = xspan[1], xspan[0]
        if flipy: yspan = yspan[1], yspan[0]

        xfn = linearMapping(xspan, self.xv)
        yfn = linearMapping(yspan, self.yv)
        return [(xfn(x),yfn(y)) for x,y in points]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectMappingMixin(object):
    def __init__(self, fromrect=None, torect=None):
        if isinstance(fromRect, (list, tuple)):
            self.fromRect = Rect(fromRect)
        elif fromrect is not None:
            self.fromrect = fromRect
        if isinstance(toRect, (list, tuple)):
            self.toRect = Rect(toRect)
        elif torect is not None:
            self.toRect = toRect

    def __repr__(self):
        return "<%s from:%r to:%r>" % (self.__class__.__name__, self.fromRect, self.toRect)

    def getMapping(self):
        return self.boxMapping(self.fromRect.getBox(), self.toRect.getBox())
    def getInverse(self):
        return self.boxMapping(self.toRect.getBox(), self.fromRect.getBox())

    def boxMapping(self, frombox, tobox):
        (x0f, y0f), (x1f, y1f) = frombox
        (x0t, y0t), (x1t, y1t) = tobox
        sx, tx = linearMapping((x0f, x1f), (x0t, x1t), False)
        sy, ty = linearMapping((y0f, y1f), (y0t, y1t), False)
        return (sx, tx), (sy,ty)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectMappin3dh(RectMappingMixin, Transform3dh):
    def asArray4x4(self):
        (sx, tx), (sy, ty) = self.getMapping()
        result = numpy.asarray([
            [sx,  0,  0, tx],
            [ 0, sy,  0, ty],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]], self.atype)
        return result

    def asInverse4x4_(self):
        (sx, tx), (sy, ty) = self.getInverse()
        result = numpy.asarray([
            [sx,  0,  0, tx],
            [ 0, sy,  0, ty],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]], self.atype)
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectMappin2dh(RectMappingMixin, Transform2dh):
    def asArray4x4(self):
        (sx, tx), (sy, ty) = self.getMapping()
        result = numpy.asarray([
            [sx,  0, tx],
            [ 0, sy, ty],
            [ 0,  0,  1]], self.atype)
        return result

    def asInverse4x4_(self):
        (sx, tx), (sy, ty) = self.getInverse()
        result = numpy.asarray([
            [sx,  0, tx],
            [ 0, sy, ty],
            [ 0,  0,  1]], self.atype)
        return result


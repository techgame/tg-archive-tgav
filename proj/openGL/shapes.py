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

from numpy import asarray, float32

from .color import ColorProperty
from .data import vertexArrays
from .raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PositionalObject(object):
    roundValues = True
    color = ColorProperty()

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    _pos = None
    def getPos(self):
        pos = self._pos
        if pos is None:
            self.setPos((0., 0., 0.))
            pos = self._pos
        return pos
    def setPos(self, pos, doUpdate=False):
        pos = asarray(pos, float32)
        if pos.shape != (3,):
            raise ValueError("Position must be a 3 elements long")
        self._pos = pos
        if doUpdate:
            self.update()
    pos = property(getPos, setPos)

    _size = None
    def getSize(self):
        size = self._size
        if size is None:
            self.setSize((0., 0., 0.))
            size = self._size
        return size
    def setSize(self, size, doUpdate=False):
        size = asarray(size, float32)
        if size.shape != (3,):
            raise ValueError("Size must be a 3 elements long")
        self._size = size
        if doUpdate:
            self.update()
    size = property(getSize, setSize)

    _align = None
    def getAlign(self):
        align = self._align
        if align is None:
            self.setAlign((0., 0., 0.))
            align = self._align
        return align
    def setAlign(self, align, doUpdate=False):
        if isinstance(align, (int, long, float)):
            align = asarray((align, align, align), float32)
        else:
            align = asarray(align, float32)

        if align.shape != (3,):
            raise ValueError("Align must be a single value, or 3 elements long")
        self._align = align
        if doUpdate:
            self.update()
    align = property(getAlign, setAlign)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Rectangle Object
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectGeometryArray(vertexArrays.VertexArray):
    drawMode = gl.GL_QUADS
    dataFormat = gl.GL_FLOAT

    @classmethod
    def fromCount(klass, count):
        return klass.fromFormat((count, 4, 3), klass.dataFormat)

    @classmethod
    def fromSingle(klass):
        return klass.fromFormat((4, 3), klass.dataFormat)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectObject(PositionalObject):
    GeometryFactory = RectGeometryArray

    def __init__(self, **kwattr):
        if kwattr: 
            self.set(kwattr)
        self.update()

    geometry = None
    def getGeometry(self, geo=None):
        if geo is None:
            geo = self.GeometryFactory.fromSingle()

        geo[:] = self.verticies()
        return geo

    def verticies(self):
        iw, ih, id = self.size
        return [[0., 0., id], [iw, 0., id], [iw, ih, id], [0., ih, id]]

    def update(self, **kwattr):
        if kwattr: 
            self.set(kwattr)

        geo = self.getGeometry(self.geometry)

        off = self.pos - (self.align*self.size)
        if self.roundValues:
            geo += off.round()
        else:
            geo += off

        self.geometry = geo
        return True

    def render(self):
        self.color.select()
        self.geometry.draw()


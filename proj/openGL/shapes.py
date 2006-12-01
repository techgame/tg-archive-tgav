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
from .data.vertexArrays import VertexArray
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

class RectArray(VertexArray):
    drawMode = gl.GL_QUADS
    dataFormat = gl.GL_FLOAT

    noFields = True
    defaultElementShape = (4, 3,)

    def getLeft(self):
        return self[..., 0, 0]
    def setLeft(self, value):
        self[..., 0, 0] = value
        self[..., 3, 0] = value
    left = property(getLeft, setLeft)

    def getRight(self):
        return self[..., 2, 0]
    def setRight(self, value):
        self[..., 1, 0] = value
        self[..., 2, 0] = value
    right = property(getRight, setRight)

    def getBottom(self):
        return self[..., 0, 1]
    def setBottom(self, value):
        self[..., 0, 1] = value
        self[..., 1, 1] = value
    bottom = property(getBottom, setBottom)

    def getTop(self):
        return self[..., 2, 1]
    def setTop(self, value):
        self[..., 2, 1] = value
        self[..., 3, 1] = value
    top = property(getTop, setTop)

    def getWidth(self):
        return self.left - self.right
    def setWidth(self, width):
        self.right = self.left + width
    width = property(getWidth, setWidth)
    
    def getHeight(self):
        return self.top - self.bottom
    def setHeight(self, height):
        self.top = self.bottom + height
    height = property(getHeight, setHeight)

    def getDims(self):
        return self.ptp(-2)
    def setDims(self, dims, align=.5):
        dims = self.newData(dims)
        self.right = self.left + dims[0]
        self.top = self.bottom + dims[1]
    dims = property(getDims, setDims)

    def getCorners(self):
        return self[..., 0:4:2, :]
    def setCorners(self, bltr):
        bltr = self.newData(bltr)
        self.left = bltr[..., 0, 0]
        self.bottom = bltr[..., 0, 1]
        self.right = bltr[..., 1, 0]
        self.top = bltr[..., 1, 1]
    corners = property(getCorners, setCorners)

    def getPosDims(self):
        corners = self.getCorners()
        return (corners[..., 0, :], (corners[...,1,:] - corners[...,0,:]))
    def setPosDims(self, pos, dims):
        items = self.newData([pos, dims])
        items[1] += items[0]
        self.setCorners(items)
    posDims = property(getPosDims, setPosDims)

    def offset(self, pos):
        self[:] += self.newData(pos)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RectGeometryArray = RectArray
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

        v = self.verticies()
        geo['v'] = v
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
            geo['v'] += off.round()
        else:
            geo['v'] += off
        self.geometry = geo
        return True

    def render(self):
        self.color.select()
        self.geometry.draw()


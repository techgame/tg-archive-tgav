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

import numpy
from numpy import asarray

from .observableData import ObservableData
from .singleArrays import Vertex as Vector

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def toAspect(size, aspect, grow=None):
    if grow is None and isinstance(aspect, tuple):
        aspect, grow = aspect

    if aspect <= 0:
        return size

    if isinstance(grow, basestring):
        if grow == 'w':
            size[0:1] = aspect * size[1:2]
            return size

        elif grow == 'h':
            size[1:2] = aspect * size[0:1]
            return size

        else:
            raise RuntimeError('Unknown grow constant %r' % (grow,))

    acurrent = float(size[0])/size[1]
    if bool(grow) ^ (aspect > acurrent):
        # new h is greater than old h
        size[1:2] = size[0:1] / aspect
    else:
        # new w is greater than old w
        size[0:1] = aspect * size[1:2]
    return size


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Rect(ObservableData):
    pos = Vector.property([0, 0, 0], propKind='aschained')
    size = Vector.property([1, 1, 0], propKind='aschained')

    #_defaultPropKind = 'astype'
    def __init__(self, rect=None, dtype=None):
        ObservableData.__init__(self)
        if rect is None:
            rect = self
        self.copyFrom(rect, dtype)

    def __repr__(self):
        name = self.__class__.__name__
        pos = self.pos
        if pos.any(): 
            pos = self.pos.tolist()
            if not pos[-1]: 
                pos = pos[:-1]
        else: pos = None

        size = self.size.tolist()
        if not size[-1]: 
            size = size[:-1]

        if pos: 
            return '%s(%s, %s)' % (name, pos, size)
        else: 
            return '%s(%s)' % (name, size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #@pos.fset
    #def _onSetPos(self, value, _paSet_):
    #    print 'onSetPos:', repr(value), _paSet_
    #    _paSet_(value)
    #    #_paSet_.fget().set(value)
    #@size.fset
    #def _onSetSize(self, value, _paSet_):
    #    print 'onSetSize:', repr(value), _paSet_
    #    _paSet_(value)
    #    #_paSet_.fget().set(value)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def tolist(self):
        return [self.pos.tolist(), self.size.tolist()]

    def getDtype(self):
        return self.size.dtype
    def setDtype(self, dtype):
        self.pos = self.pos.astype(dtype)
        self.size = self.size.astype(dtype)
    dtype = property(getDtype, setDtype)

    def astype(self, dtype):
        return self.copy(dtype)

    def __copy__(self):
        return self.copy()

    def copy(self, dtype=None):
        r = self.__new__(self.__class__)
        return r.copyFrom(self, dtype)

    def copyFrom(self, other, dtype=None):
        if dtype is None:
            dtype = self.dtype

        self.pos = other.pos.astype(dtype)
        self.size = other.size.astype(dtype)
        return self

    #~ setters and construction methods ~~~~~~~~~~~~~~~~~

    def setRect(self, rect, aspect=None, align=None, dtype=None):
        self.pos.set(rect.pos)
        self.size.set(rect.size)
        self.setAspect(aspect, align)
        return self
    def fromRect(self, rect, aspect=None, align=None, dtype=None):
        self = klass(rect, dtype)
        self.setAspect(aspect, align)
        return self

    @classmethod
    def fromSize(klass, size, aspect=None, align=None, dtype=None):
        self = klass(dtype=dtype)
        self.setSize(size, aspect, align)
        return self

    @classmethod
    def fromPosSize(klass, pos, size, aspect=None, align=None, dtype=None):
        self = klass(dtype=dtype)
        self.setPosSize(pos, size, aspect, align)
        return self

    def setPosSize(self, pos, size, aspect=None, align=None):
        self.pos.set(pos)
        self.setSize(size, aspect, align)
        return self

    @classmethod
    def fromCorners(klass, p0, p1, dtype=None):
        self = klass(dtype=dtype)
        self.setCorners(p0, p1)
        return self

    def setCorners(self, p0, p1):
        pv = asarray([p0, p1], self.pos.dtype)
        pos = pv.min(0)
        size = pv.max(0) - pos
        return self.setPosSize(pos, size)

    def centerIn(self, other): 
        return self.alignIn(.5, other)
    def alignIn(self, align, other):
        if isinstance(other, Rect):
            self.pos.set(other.pos + align*(other.size-self.size))
        else: 
            self.pos.set(self.pos + align*(other-self.size))
        return self

    def getCorner(self):
        return self.pos + self.size
    corner = property(getCorner)
    
    def setSize(self, size, aspect=None, align=None):
        self.size.set(numpy.abs(size))

        if aspect is not None or align is not None:
            return self.setAspect(aspect, align)

        return self

    def sizeAs(self, aspect):
        return self.toAspect(self.size.copy(), aspect)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getAspect(self):
        s = self.size[0:2].astype(float)
        return s[0]/s[1]
    def setAspect(self, aspect, align=None):
        if aspect is None: return self

        if align is None:
            self.toAspect(self.size, aspect)
            return self
        else:
            size = self.size.copy()
            self.toAspect(self.size, aspect)

            return self.alignIn(align, size)
    aspect = property(getAspect, setAspect)

    toAspect = staticmethod(toAspect)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Named accessors
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getWidth(self): return self.size[0]
    def setWidth(self, width): self.size[0] = max(0, width)
    width = property(getWidth, setWidth)

    def getHeight(self): return self.size[1]
    def setHeight(self, height): self.size[1] = max(0, height)
    height = property(getHeight, setHeight)

    def getDepth(self): return self.size[2]
    def setDepth(self, depth): self.size[2] = max(0, depth)
    depth = property(getDepth, setDepth)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getLeft(self): return self.pos[0]
    def setLeft(self, left): self.pos[0] = left
    left = property(getLeft, setLeft)

    def getBottom(self): return self.pos[1]
    def setBottom(self, bottom): self.pos[1] = bottom
    bottom = property(getBottom, setBottom)

    def getFront(self): return self.pos[2]
    def setFront(self, front): self.pos[2] = front
    front = property(getFront, setFront)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getRight(self): return self.pos[0] + self.size[0]
    def setRight(self, right): self.pos[0] = right - self.size[0]
    right = property(getRight, setRight)

    def getTop(self): return self.pos[1] + self.size[1]
    def setTop(self, top): self.pos[1] = top - self.size[1]
    top = property(getTop, setTop)
    
    def getBack(self): return self.pos[2] + self.size[2]
    def setBack(self, back): self.pos[2] = back - self.size[2]
    back = property(getBack, setBack)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Recti(Rect):
    pos = Vector.property([0, 0, 0], dtype='i', propKind='aschained')
    size = Vector.property([1, 1, 0], dtype='i', propKind='aschained')

class Rectf(Rect):
    pos = Vector.property([0, 0, 0], dtype='f', propKind='aschained')
    size = Vector.property([1, 1, 0], dtype='f', propKind='aschained')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = ['Rect', 'Recti', 'Rectf', 'toAspect']


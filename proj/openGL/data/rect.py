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

from numpy import asarray

from .observableData import ObservableData
from .singleArrays import Vector

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
    _pos = Vector([0, 0, 0], 'f')
    _size = Vector([1, 1, 0], 'f')

    defaultPropKind = 'asType'
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

    def getDtype(self):
        dtype = self._size.dtype
        assert dtype == self._pos.dtype, (dtype, self._pos.dtype)
        return dtype
    def setDtype(self, dtype):
        self._pos = self._pos.astype(dtype)
        self._size = self._size.astype(dtype)
        self._kvnotify_('set', 'pos')
        self._kvnotify_('set', 'size')
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
        self._pos = other._pos.astype(dtype)
        self._size = other._size.astype(dtype)
        return self

    #~ setters and construction methods ~~~~~~~~~~~~~~~~~

    def setRect(self, rect, aspect=None, align=None, dtype=None):
        self.copyFrom(rect)
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
        self.pos = pos
        self.setSize(size, aspect, align)
        return self

    @classmethod
    def fromCorners(klass, p0, p1, dtype=None):
        self = klass(dtype=dtype)
        self.setCorners(p0, p1)
        return self

    def setCorners(self, p0, p1):
        pv = asarray([p0, p1], self._pos.dtype)
        pos = pv.min(0)
        size = pv.max(0) - pos
        return self.setPosSize(pos, size)

    def getPos(self): 
        return self._pos.copy()
    def setPos(self, pos):
        self._pos.set(pos)
        self._kvnotify_('set', 'pos')
        return self
    pos = property(getPos, setPos)

    def centerIn(self, other): 
        return self.alignIn(.5, other)
    def alignIn(self, align, other):
        if isinstance(other, Rect):
            self.pos = other.pos + align*(other.size-self.size) 
        else: self.pos += align*(other-self.size) 
        return self

    def getCorner(self):
        return self._pos + self._size
    corner = property(getCorner)
    
    def getSize(self):
        return self._size.copy()
    def setSize(self, size, aspect=None, align=None):
        selfSize = self._size
        selfSize.set(size)
        selfSize[selfSize < 0] = 0

        if aspect is not None or align is not None:
            return self.setAspect(aspect, align)

        self._kvnotify_('set', 'size')
        return self
    size = property(getSize, setSize)

    def sizeAs(self, aspect):
        return self.toAspect(self._size.copy(), aspect)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getAspect(self):
        s = self._size[0:2].astype(float)
        return s[0]/s[1]
    def setAspect(self, aspect, align=None):
        if aspect is None: return self

        if align is None:
            self.toAspect(self._size, aspect)
            self._kvnotify_('set', 'size')
            return self
        else:
            size = self._size.copy()
            self.toAspect(self._size, aspect)
            self._kvnotify_('set', 'size')

            return self.alignIn(align, size)
    aspect = property(getAspect, setAspect)

    toAspect = staticmethod(toAspect)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getWidth(self): 
        return self._size[0]
    def setWidth(self, width): 
        self._size[0] = max(0, width)
        self._kvnotify_('set', 'size')
    width = property(getWidth, setWidth)

    def getHeight(self): 
        return self._size[1]
    def setHeight(self, height): 
        self._size[1] = max(0, height)
        self._kvnotify_('set', 'size')
    height = property(getHeight, setHeight)

    def getDepth(self): 
        return self._size[2]
    def setDepth(self, depth): 
        self._size[2] = max(0, depth)
        self._kvnotify_('set', 'size')
    depth = property(getDepth, setDepth)
    
    def getLeft(self): 
        return self._pos[0]
    def setLeft(self, left):
        self._pos[0] = left
        self._kvnotify_('set', 'pos')
    left = property(getLeft, setLeft)

    def getBottom(self): 
        return self._pos[1]
    def setBottom(self, bottom):
        self._pos[1] = bottom
        self._kvnotify_('set', 'pos')
    bottom = property(getBottom, setBottom)

    def getFront(self): 
        return self._pos[2]
    def setFront(self, front):
        self._pos[2] = front
        self._kvnotify_('set', 'pos')
    front = property(getFront, setFront)

    def getRight(self): 
        return self._pos[0] + self._size[0]
    def setRight(self, right):
        self._size[0] = right - self._pos[0]
        self._kvnotify_('set', 'size')
    right = property(getRight, setRight)

    def getTop(self): 
        return self._pos[1] + self._size[1]
    def setTop(self, top):
        self._size[1] = top - self._pos[0]
        self._kvnotify_('set', 'size')
    top = property(getTop, setTop)
    
    def getBack(self): 
        return self._pos[2] + self._size[2]
    def setBack(self, back):
        self._size[2] = back - self._pos[2]
        self._kvnotify_('set', 'size')
    back = property(getBack, setBack)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Recti(Rect):
    _pos = Vector([0, 0, 0], 'i')
    _size = Vector([1, 1, 0], 'i')

class Rectf(Rect):
    _pos = Vector([0, 0, 0], 'f')
    _size = Vector([1, 1, 0], 'f')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = ['Rect', 'Recti', 'Rectf', 'toAspect']


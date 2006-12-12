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

from .singleArrays import Vector

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def toAspect(size, aspect, grow=False):
    if aspect <= 0:
        return size
    acurrent = size[0]/size[1]
    if grow ^ (aspect > acurrent):
        # new h is greater than old h
        size[1:2] = size[0:1] / aspect
    else:
        # new w is greater than old w
        size[0:1] = aspect * size[1:2]
    return size

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectBasic(object):
    _pos = Vector([0., 0., 0.], 'f')
    _size = Vector([1., 1., 0.], 'f')

    def __init__(self, *args, **kw):
        self.set(*args, **kw)

    def __repr__(self):
        name = self.__class__.__name__
        v0 = self.v0
        if v0.any(): 
            v0 = self.v0.tolist()
            if not v0[-1]: 
                v0 = v0[:-1]
        else: v0 = None

        size = self.size.tolist()
        if not size[-1]: 
            size = size[:-1]

        if v0: 
            return '%s(%s, %s)' % (name, v0, size)
        else: 
            return '%s(%s)' % (name, size)

    def copy(self):
        r = self.__new__(self.__class__)
        return r.copyFrom(self)
        new = self.new()
        new._pos = self._pos.copy()
        new._size = self._size.copy()
        return new

    def copyFrom(self, other):
        self._pos = other._pos.copy()
        self._size = other._size.copy()
        return self

    def set(self, *args, **kw):
        dtype = None
        aspect = None

        if len(args) == 1:
            size, = args
            if isinstance(size, RectBasic):
                pos = size.pos.copy()
                dtype = pos.dtype
                size = size.size.copy()
            if len(size) <= 3:
                v0 = None
            elif len(size) == 4:
                v0, size = size[:2], size[2:]
            elif len(size) == 6:
                v0, size = size[:3], size[3:]

            v1 = None

        elif len(args) == 2:
            v0, size = args
            v1 = None

        else:
            v0 = kw.pop('v0', None)
            v1 = kw.pop('v1', None)
            size = kw.pop('size', None)

        dtype = kw.pop('dtype', dtype)
        aspect = kw.pop('aspect', aspect)

        if kw:
            raise Exception("Unexpected arguments: %s" % (kw.keys(),))

        self._pos = Vector(self._pos, dtype)
        self._size = Vector(self._size, dtype)

        if v0 is not None:
            self.v0 = v0
        if v1 is not None:
            self.v1 = v1
        if size is not None:
            self.size = size
        if aspect is not None:
            self.aspect = aspect

        return self

    def _kvnotify_(self, op, key): 
        """This method is intended to be replaced by a mixin with ObservableObject"""

    def getV0(self): 
        return self._pos.copy()
    def setV0(self, v0):
        self._pos.set(v0)
        self._kvnotify_('set', 'pos')
        return self
    pos = v0 = property(getV0, setV0)

    def getV1(self):
        return self._pos + self._size
    def setV1(self, v1):
        return self.setSize(v1 - self._pos[:len(v1)])
    v1 = property(getV1, setV1)
    
    def getSize(self):
        return self._size.copy()
    def setSize(self, size):
        selfSize = self._size
        selfSize.set(size)
        selfSize[selfSize < 0] = 0

        self._kvnotify_('set', 'size')
        return self
    size = property(getSize, setSize)

    def sizeAs(self, aspect, grow=False):
        return self.toAspect(self._size.copy(), aspect, grow)
    
    def getAspect(self):
        s = self._size
        return s[0]/s[1]
    def setAspect(self, aspect, grow=False):
        self.toAspect(self._size, aspect, grow)
        self._kvnotify_('set', 'size')
        return self
    aspect = property(getAspect, setAspect)

    toAspect = staticmethod(toAspect)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RectSidesMixin(object):
    _isBottomLeft = True

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
        if self._isBottomLeft:
            return self._pos[1]
        else: return self._v1[1]
    def setBottom(self, bottom):
        if self._isBottomLeft:
            self._pos[1] = bottom
            self._kvnotify_('set', 'pos')
        else: 
            self._v1[1] = bottom
            self._kvnotify_('set', 'size')
    bottom = property(getBottom, setBottom)

    def getFront(self): 
        return self._pos[2]
    def setFront(self, front):
        self._pos[2] = front
        self._kvnotify_('set', 'pos')
    front = property(getFront, setFront)

    def getRight(self): 
        return self._v1[0]
    def setRight(self, right):
        self._v1[0] = right
        self._kvnotify_('set', 'size')
    right = property(getRight, setRight)

    def getTop(self): 
        if self._isBottomLeft:
            return self._v1[1]
        else: return self._pos[1]
    def setTop(self, top):
        if self._isBottomLeft:
            self._v1[1] = top
            self._kvnotify_('set', 'size')
        else:
            self._pos[1] = top
            self._kvnotify_('set', 'pos')
    top = property(getTop, setTop)
    
    def getBack(self): 
        return self._v1[2]
    def setBack(self, back):
        self._v1[2] = back
        self._kvnotify_('set', 'size')
    back = property(getBack, setBack)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FlexRect(RectSidesMixin, RectBasic):
    """Flexible Rect implementation
    
    Can be changed from bottom left to top left via _isBottomLeft attribute.
    GLRect and WRect override bottom and top to optimize for their setting of _isBottomLeft attribute.
    """
    _isBottomLeft = True

class BRect(FlexRect):
    """BRect implementation.
    
    Optimised for bottom left configuration"""
    _isBottomLeft = True

    def getBottom(self): 
        return self._pos[1]
    def setBottom(self, bottom):
        self._pos[1] = bottom
        self._kvnotify_('set', 'pos')
    bottom = property(getBottom, setBottom)

    def getTop(self): 
        return self._v1[1]
    def setTop(self, top):
        self._v1[1] = top
        self._kvnotify_('set', 'size')
    top = property(getTop, setTop)

Rect = BRect
GLRect = BRect
    
class TRect(FlexRect):
    """TRect implementation.
    
    Optimised for top left configuration"""
    _isBottomLeft = False

    def getBottom(self): 
        return self._v1[1]
    def setBottom(self, bottom):
        self._v1[1] = bottom
        self._kvnotify_('set', 'size')
    bottom = property(getBottom, setBottom)

    def getTop(self): 
        return self._pos[1]
    def setTop(self, top):
        self._pos[1] = top
        self._kvnotify_('set', 'pos')
    top = property(getTop, setTop)
WRect = TRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = ['Rect', 'FlexRect', 'BRect', 'TRect', 'GLRect', 'WRect']


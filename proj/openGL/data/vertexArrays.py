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
from numpy import shape, ndarray, dot, asarray, dstack

from .glArrayBase import GLArrayBase
from .glArrayDataType import GLArrayDataType
from .colorFormats import ColorFormatMixin

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Utility functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def blend(u0, u1, a):
    return dot(dstack([u0, u1]), asarray([1-a, a]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Data Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AtIndexProperty(object):
    def __init__(self, idx):
        self.idx = idx
    def __get__(self, obj, klass):
        if obj is None:
            return self
        else:
            return obj[..., self.idx]
    def __set__(self, obj, value):
        obj[..., self.idx] = value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DataArrayBase(GLArrayBase):
    __array_priority__ = 25.0
    _defaultPropKind = 'asarraytype'

    gldtype = GLArrayDataType()

    def blend(self, other, alpha, copy=True):
        r = blend(self, other, alpha)
        if copy: 
            return r.astype(self.dtype)

        self[:] = r
        return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VectorArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('bhlifd', (1,2,3,4))
    gldtype.setDefaultFormat('3f')
    glinfo = gldtype.arrayInfoFor('vector')

    x = AtIndexProperty(0)
    y = AtIndexProperty(1)
    z = AtIndexProperty(2)

class VertexArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (2,3,4))
    gldtype.setDefaultFormat('3f')
    glinfo = gldtype.arrayInfoFor('vertex')

    x = AtIndexProperty(0)
    y = AtIndexProperty(1)
    z = AtIndexProperty(2)

class TextureCoordArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (1,2,3,4))
    gldtype.setDefaultFormat('3f')
    glinfo = gldtype.arrayInfoFor('texture_coord')

    r = AtIndexProperty(0)
    s = AtIndexProperty(1)
    t = AtIndexProperty(2)
TexCoordArray = TextureCoordArray

class MultiTextureCoordArray(TextureCoordArray):
    gldtype = TextureCoordArray.gldtype.copy()
    glinfo = gldtype.arrayInfoFor('multi_texture_coord')
MultiTexCoordArray = MultiTextureCoordArray

class NormalArray(DataArrayBase):
    default = asarray([0, 0, 1], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('bhlifd', (3,))
    gldtype.setDefaultFormat('3f')
    glinfo = gldtype.arrayInfoFor('normal')

class ColorArray(ColorFormatMixin, DataArrayBase):
    default = asarray([1., 1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,4))
    gldtype.setDefaultFormat('4f')
    glinfo = gldtype.arrayInfoFor('color')

    r = AtIndexProperty(0)
    g = AtIndexProperty(1)
    b = AtIndexProperty(2)
    a = AtIndexProperty(3)

class SecondaryColorArray(ColorArray):
    default = asarray([1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,))
    gldtype.setDefaultFormat('3f')
    glinfo = gldtype.arrayInfoFor('secondary_color')

    r = AtIndexProperty(0)
    g = AtIndexProperty(1)
    b = AtIndexProperty(2)

class ColorIndexArray(DataArrayBase):
    default = asarray([0], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('Bhlifd', (1,))
    gldtype.setDefaultFormat('1B')
    glinfo = gldtype.arrayInfoFor('color_index')

class FogCoordArray(DataArrayBase):
    default = asarray([0.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('fd', (1,))
    gldtype.setDefaultFormat('1f')
    glinfo = gldtype.arrayInfoFor('fog_coord')

class EdgeFlagArray(DataArrayBase):
    default = asarray([1], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('B', (1,))
    gldtype.setDefaultFormat('1B')
    glinfo = gldtype.arrayInfoFor('edge_flag')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if isinstance(value, type) and issubclass(value, GLArrayBase))
__all__.append('blend')


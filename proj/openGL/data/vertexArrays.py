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

class DataArrayBase(GLArrayBase):
    gldtype = GLArrayDataType()

    def blend(self, other, alpha, copy=True):
        r = blend(self, other, alpha)
        if copy: 
            return r.astype(self.dtype)

        self[:] = r
        return self

    def get(self, at=Ellipsis):
        return self[at]
    def set(self, data, at=Ellipsis, fill=0):
        l = shape(data)
        if not l:
            # fill with data
            self[at] = data
        else:
            l = min(l[-1], self.shape[-1])
            if isinstance(data, ndarray):
                self[at,:l] = data[at, :l]
            else:
                self[at,:l] = data[:l]
            self[at,l:] = fill
        return self
    def setPart(self, data, at=Ellipsis):
        l = shape(data)
        if not l:
            # fill with data
            self[at] = data
        else:
            l = min(l[-1], self.shape[-1])
            self[at,:l] = data[:l]
        return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VectorArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('vector')

class VertexArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('vertex')

class TextureCoordArray(DataArrayBase):
    default = asarray([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('texture_coord')

class MultiTextureCoordArray(TextureCoordArray):
    gldtype = TextureCoordArray.gldtype.copy()
    glinfo = gldtype.arrayInfoFor('multi_texture_coord')

class NormalArray(DataArrayBase):
    default = asarray([0, 0, 1], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('bhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('normal')

class ColorArray(ColorFormatMixin, DataArrayBase):
    default = asarray([1., 1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,4), default='4f')
    glinfo = gldtype.arrayInfoFor('color')

class SecondaryColorArray(ColorArray):
    default = asarray([1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('secondary_color')

class ColorIndexArray(DataArrayBase):
    default = asarray([0], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('Bhlifd', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('color_index')

class FogCoordArray(DataArrayBase):
    default = asarray([0.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('fd', (1,), default='1f')
    glinfo = gldtype.arrayInfoFor('fog_coord')

class EdgeFlagArray(DataArrayBase):
    default = asarray([1], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('B', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('edge_flag')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if isinstance(value, type) and issubclass(value, GLArrayBase))
__all__.append('blend')


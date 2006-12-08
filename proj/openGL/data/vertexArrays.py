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

from .glArrayBase import GLArrayBase
from .glArrayDataType import GLArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Data Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DataArrayBase(GLArrayBase):
    gldtype = GLArrayDataType()

    def get(self, at=Ellipsis):
        return self[at]
    def set(self, data, at=Ellipsis, fill=0):
        l = numpy.shape(data)[-1]
        self[at,:l] = data
        self[at,l:] = fill
        return self
    def setPart(self, data, at=Ellipsis):
        l = numpy.shape(data)[-1]
        self[at,:l] = data
        return self

class VertexArray(DataArrayBase):
    default = numpy.array([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('vertex')

class TextureCoordArray(DataArrayBase):
    default = numpy.array([0], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')
    glinfo = gldtype.arrayInfoFor('texture_coord')

class MultiTextureCoordArray(TextureCoordArray):
    gldtype = TextureCoordArray.gldtype.copy()
    glinfo = gldtype.arrayInfoFor('multi_texture_coord')

class NormalArray(DataArrayBase):
    default = numpy.array([0, 0, 1], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('bhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('normal')

class ColorArray(DataArrayBase):
    default = numpy.array([1., 1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,4), default='4f')
    glinfo = gldtype.arrayInfoFor('color')

class SecondaryColorArray(DataArrayBase):
    default = numpy.array([1., 1., 1.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('BHLIbhlifd', (3,), default='3f')
    glinfo = gldtype.arrayInfoFor('secondary_color')

class ColorIndexArray(DataArrayBase):
    default = numpy.array([0], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('Bhlifd', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('color_index')

class FogCoordArray(DataArrayBase):
    default = numpy.array([0.], 'f')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('fd', (1,), default='1f')
    glinfo = gldtype.arrayInfoFor('fog_coord')

class EdgeFlagArray(DataArrayBase):
    default = numpy.array([1], 'B')

    gldtype = GLArrayDataType()
    gldtype.addFormatGroups('B', (1,), default='1B')
    glinfo = gldtype.arrayInfoFor('edge_flag')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__all__ = sorted(name for name, value in vars().items() if isinstance(value, type) and issubclass(value, GLArrayBase))


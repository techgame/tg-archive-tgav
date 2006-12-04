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
from ..raw import gl
from .glArrayBase import GLArrayBase

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBase(GLArrayBase):
    gldtype = GLArrayBase.gldtype.copy()

    def get(self, at=Ellipsis):
        return self[at]
    def set(self, data, at=Ellipsis):
        l = numpy.shape(data)[-1]
        self[at,:l] = data
        return self

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(ArrayBase):
    defaultValue = numpy.array([0], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_VERTEX_ARRAY
    gldtype.addFormatGroups('hlifd', (2,3,4), default='3f')

class TexureCoordArray(ArrayBase):
    defaultValue = numpy.array([0], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_TEXTURE_COORD_ARRAY
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')

class NormalArray(ArrayBase):
    defaultValue = numpy.array([0, 0, 1], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_NORMAL_ARRAY
    gldtype.addFormatGroups('bhlifd', (3,), default='3f')

class ColorArray(ArrayBase):
    defaultValue = numpy.array([1., 1., 1., 1.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_COLOR_ARRAY
    gldtype.addFormatGroups('BHIbhlifd', (3,4), default='4f')

class SecondaryColorArray(ArrayBase):
    defaultValue = numpy.array([1., 1., 1.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_SECONDARY_COLOR_ARRAY
    gldtype.addFormatGroups('BHIbhlifd', (3,), default='3f')

class ColorIndexArray(ArrayBase):
    defaultValue = numpy.array([0], 'B')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_INDEX_ARRAY
    gldtype.addFormatGroups('Bhlifd', (1,), default='1B')

class FogCoordArray(ArrayBase):
    defaultValue = numpy.array([0.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_FOG_COORD_ARRAY
    gldtype.addFormatGroups('fd', (1,), default='1f')

class EdgeFlagArray(ArrayBase):
    defaultValue = numpy.array([1], 'B')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.glBufferType = gl.GL_EDGE_FLAG_ARRAY
    gldtype.addFormatGroups('B', (1,), default='1B')


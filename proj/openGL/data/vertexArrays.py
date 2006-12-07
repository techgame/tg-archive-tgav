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
    default = numpy.array([0], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('vertex')
    gldtype.addFormatGroups('hlifd', (2,3,4), default='3f')

class TexureCoordArray(ArrayBase):
    default = numpy.array([0], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('texture_coord')
    gldtype.addFormatGroups('hlifd', (1,2,3,4), default='3f')

class MultiTexureCoordArray(TexureCoordArray):
    gldtype = TexureCoordArray.gldtype.copy()
    gldtype.setKind('multi_texture_coord')

class NormalArray(ArrayBase):
    default = numpy.array([0, 0, 1], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('normal')
    gldtype.addFormatGroups('bhlifd', (3,), default='3f')

class ColorArray(ArrayBase):
    default = numpy.array([1., 1., 1., 1.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('color')
    gldtype.addFormatGroups('BHLIbhlifd', (3,4), default='4f')

class SecondaryColorArray(ArrayBase):
    default = numpy.array([1., 1., 1.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('secondary_color')
    gldtype.addFormatGroups('BHLIbhlifd', (3,), default='3f')

class ColorIndexArray(ArrayBase):
    default = numpy.array([0], 'B')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('color_index')
    gldtype.addFormatGroups('Bhlifd', (1,), default='1B')

class FogCoordArray(ArrayBase):
    default = numpy.array([0.], 'f')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('fog_coord')
    gldtype.addFormatGroups('fd', (1,), default='1f')

class EdgeFlagArray(ArrayBase):
    default = numpy.array([1], 'B')

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.setKind('edge_flag')
    gldtype.addFormatGroups('B', (1,), default='1B')


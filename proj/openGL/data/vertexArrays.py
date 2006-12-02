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
    defaultValue = 0

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('hifd', (2,3,4), default='3f')

class TexureCoordArray(ArrayBase):
    defaultValue = 0

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('hifd', (1,2,3,4), default='3f')

class NormalArray(ArrayBase):
    defaultValue = (0, 0, 1)

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('bhifd', (3,), default='3f')

class ColorArray(ArrayBase):
    defaultValue = 1

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('BHIbhifd', (3,4), default='4f')

class SecondaryColorArray(ArrayBase):
    defaultValue = 1

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('BHIbhifd', (3,), default='3f')

class ColorIndexArray(ArrayBase):
    defaultValue = 0

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('Bhifd', (1,), default='1B')

class FogCoordArray(ArrayBase):
    defaultValue = 0

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('fd', (1,), default='1f')

class EdgeFlagArray(ArrayBase):
    defaultValue = 1

    gldtype = GLArrayBase.gldtype.copy()
    gldtype.addFormats('B', (1,), default='1B')


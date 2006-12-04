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

from TG.openGL.data.glArrayDataType import GLInterleavedArrayDataType
from TG.openGL.data.vertexArrays import GLArrayBase

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Interleaved Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FieldProperty(object):
    def __init__(self, fieldName):
        self.fieldName = fieldName
    def __get__(self, obj, klass):
        if obj is None:
            return self
        else:
            return obj[self.fieldName]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InterleavedArrays(GLArrayBase):
    defaultValue = numpy.array([0], 'f')
    gldtype = GLInterleavedArrayDataType()
    gldtype.setDefaultFormat(gl.GL_V3F)

    vertex = v = FieldProperty('v')
    colors = c = FieldProperty('c')
    normals = n = FieldProperty('n')
    texcoords = tex = t = FieldProperty('t')

    glInterleavedArrays = staticmethod(gl.glInterleavedArrays)

    def enable(self):
        pass

    def bind(self, vboOffset=None):
        if vboOffset is None:
            self.glInterleavedArrays(self.dataFormat, self.strides[-1], self.ctypes)
        else:
            self.glInterleavedArrays(self.dataFormat, self.strides[-1], vboOffset)

    def disable(self):
        pass

    def unbind(self):
        self.glInterleavedArrays(self.dataFormat, 0, None)

    def draw(self, drawMode=None, vboOffset=None):
        self.select(vboOffset)
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)
    def drawRaw(self, drawMode=None):
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)


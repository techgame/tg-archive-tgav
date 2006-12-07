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
from .glArrayDataType import GLInterleavedArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Interleaved Data Array
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FieldProperty(object):
    def __init__(self, fieldName):
        self.fieldName = fieldName
    def __get__(self, obj, klass):
        if obj is None:
            return self
        else:
            return obj[self.fieldName]
    def __set__(self, obj, value):
        obj[self.fieldName][:] = value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InterleavedArray(GLArrayBase):
    defaultValue = numpy.array([0], 'f')
    gldtype = GLInterleavedArrayDataType()
    gldtype.setDefaultFormat(gl.GL_V3F)
    glinfo = gldtype.arrayInfoFor('interleaved')

    vertex = v = FieldProperty('v')
    colors = c = FieldProperty('c')
    normals = n = FieldProperty('n')
    texcoords = tex = t = FieldProperty('t')


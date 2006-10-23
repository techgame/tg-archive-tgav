##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy
from OpenGL import GL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBase(object):
    """VertexArray extension objects."""

    atype = numpy.Float32 

    def __init__(self, data, atype=None):
        if isinstance(data, numpy.ArrayType):
            self.data = data
        else:
            self.data = numpy.asarray(data, atype or self.atype)

    def glSelect(self, context):
        context.ClientStateMgr.Enable(self.glArrayType)
        self.glArrayCall(self.data)
    GLExecute = GLSelect

    def glDeselect(self, context):
        context.ClientStateMgr.Disable(self.glArrayType)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(ArrayBase):
    glArrayType = GL.GL_VERTEX_ARRAY
    glArrayCall = GL.VertexArray

class NormalArray(ArrayBase):
    glArrayType = GL.GL_NORMAL_ARRAY
    glArrayCall = GL.NormalArray

class TexCoordArray(ArrayBase):
    glArrayType = GL.GL_TEXTURE_COORD_ARRAY
    glArrayCall = GL.TexCoordArray

class ColorArray(ArrayBase):
    glArrayType = GL.GL_COLOR_ARRAY
    glArrayCall = GL.ColorArray

class EdgeFlagArray(ArrayBase):
    atype = numpy.UInt8
    glArrayType = GL.GL_EDGE_FLAG_ARRAY
    _glArrayCall = GL.EdgeFlagArray


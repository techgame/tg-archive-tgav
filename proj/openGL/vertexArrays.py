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

from functools import partial

from numpy import ndarray, array

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBase(ndarray):
    dataFormat = None
    dataFormatMap = dict(
        uint8=gl.GL_UNSIGNED_BYTE,
        uint16=gl.GL_UNSIGNED_SHORT,
        uint32=gl.GL_UNSIGNED_INT,

        int8=gl.GL_BYTE,
        int16=gl.GL_SHORT,
        int32=gl.GL_INT,

        float32=gl.GL_FLOAT,
        float64=gl.GL_DOUBLE,
        )

    def __new__(klass, data, dtype=None, copy=False):
        data = array(data, dtype, copy=copy, order='C', ndmin=1)
        return data.view(klass)

    def __init__(self, data=None, dtype=None, copy=False):
        self._config()

    def _config(self):
        self._as_parameter_ = self.ctypes._as_parameter_
        self._inferDataFormat()

    def _inferDataFormat(self):
        self.dataFormat = self.dataFormatMap[self.dtype.name]

    def select(self):
        self.enable()
        self.bind()

    def enable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def glArrayPointer(self, count, dataFormat, stride, ptr):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def bind(self):
        self.glArrayPointer(len(self), self.dataFormat, 0, self)

    def deselect(self):
        self.unbind()
        self.disable()

    def disable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def unbind(self):
        self.glArrayPointer(0, self.dataFormat, 0, None)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(ArrayBase):
    glArrayType = gl.GL_VERTEX_ARRAY
    glArrayPointer = staticmethod(gl.glVertexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class TexCoordArray(ArrayBase):
    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexCoordArray(TexCoordArray):
    glClientActiveTexture = staticmethod(gl.glClientActiveTexture)
    texUnit = gl.GL_TEXTURE0

    def bind(self):
        self.glClientActiveTexture(self.texUnit)
        self.glArrayPointer(len(self), self.dataFormat, 0, self)
    def unbind(self):
        self.glClientActiveTexture(self.texUnit)
        self.glArrayPointer(0, self.dataFormat, 0, None)

class NormalArray(ArrayBase):
    glArrayType = gl.GL_NORMAL_ARRAY
    glArrayPointer = staticmethod(gl.glNormalPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorArray(ArrayBase):
    glArrayType = gl.GL_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class SecondaryColorArray(ArrayBase):
    glArrayType = gl.GL_SECONDARY_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glSecondaryColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorIndexArray(ArrayBase):
    glArrayType = gl.GL_INDEX_ARRAY
    glArrayPointer = staticmethod(gl.glIndexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class FogCoordArray(ArrayBase):
    glArrayType = gl.GL_FOG_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glFogCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class EdgeFlagArray(ArrayBase):
    glArrayType = gl.GL_EDGE_FLAG_ARRAY
    glArrayPointer = staticmethod(gl.glEdgeFlagPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Interleaved Arrays
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class InterleavedArrays(ArrayBase):
    dataFormat = None
    dataFormatMap = dict()

    glInterleavedArrays = staticmethod(gl.glInterleavedArrays)

    def enable(self):
        pass

    def bind(self):
        self.glInterleavedArrays(self.dataFormat, 0, self)

    def disable(self):
        pass

    def unbind(self):
        self.glInterleavedArrays(self.dataFormat, 0, None)


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

from numpy import ndarray, array, dtype

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

    def __new__(klass, data=[], dtype=None, copy=False):
        data = array(data, dtype, copy=copy, order='C', ndmin=1)
        return data.view(klass)

    def __init__(self, data=[], dtype=None, copy=False):
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

class TexureCoordArray(ArrayBase):
    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexureCoordArray(TexureCoordArray):
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

def gldtype(gltypestr):
    parts = [[i.strip() for i in p.split(':', 1)] for p in gltypestr.split(',')]
    names = [e[0] for e in parts]
    formats = [e[1] for e in parts]
    return dtype(dict(names=names, formats=formats))

class InterleavedArrays(ArrayBase):
    dataFormat = None
    dataFormatMap = {
        gl.GL_V2F: gl.GL_V2F,
        'V2F': gl.GL_V2F,
        'v2f': gl.GL_V2F,
        gl.GL_V3F: gl.GL_V3F,
        'V3F': gl.GL_V3F,
        'v3f': gl.GL_V3F,
        gl.GL_C4UB_V2F: gl.GL_C4UB_V2F,
        'C4UB_V2F': gl.GL_C4UB_V2F,
        'c4ub_v2f': gl.GL_C4UB_V2F,
        gl.GL_C4UB_V3F: gl.GL_C4UB_V3F,
        'C4UB_V3F': gl.GL_C4UB_V3F,
        'c4ub_v3f': gl.GL_C4UB_V3F,
        gl.GL_C3F_V3F: gl.GL_C3F_V3F,
        'C3F_V3F': gl.GL_C3F_V3F,
        'c3f_v3f': gl.GL_C3F_V3F,
        gl.GL_N3F_V3F: gl.GL_N3F_V3F,
        'N3F_V3F': gl.GL_N3F_V3F,
        'n3f_v3f': gl.GL_N3F_V3F,
        gl.GL_C4F_N3F_V3F: gl.GL_C4F_N3F_V3F,
        'C4F_N3F_V3F': gl.GL_C4F_N3F_V3F,
        'c4f_n3f_v3f': gl.GL_C4F_N3F_V3F,
        gl.GL_T2F_V3F: gl.GL_T2F_V3F,
        'T2F_V3F': gl.GL_T2F_V3F,
        't2f_v3f': gl.GL_T2F_V3F,
        gl.GL_T4F_V4F: gl.GL_T4F_V4F,
        'T4F_V4F': gl.GL_T4F_V4F,
        't4f_v4f': gl.GL_T4F_V4F,
        gl.GL_T2F_C4UB_V3F: gl.GL_T2F_C4UB_V3F,
        'T2F_C4UB_V3F': gl.GL_T2F_C4UB_V3F,
        't2f_c4ub_v3f': gl.GL_T2F_C4UB_V3F,
        gl.GL_T2F_C3F_V3F: gl.GL_T2F_C3F_V3F,
        'T2F_C3F_V3F': gl.GL_T2F_C3F_V3F,
        't2f_c3f_v3f': gl.GL_T2F_C3F_V3F,
        gl.GL_T2F_N3F_V3F: gl.GL_T2F_N3F_V3F,
        'T2F_N3F_V3F': gl.GL_T2F_N3F_V3F,
        't2f_n3f_v3f': gl.GL_T2F_N3F_V3F,
        gl.GL_T2F_C4F_N3F_V3F: gl.GL_T2F_C4F_N3F_V3F,
        'T2F_C4F_N3F_V3F': gl.GL_T2F_C4F_N3F_V3F,
        't2f_c4f_n3f_v3f': gl.GL_T2F_C4F_N3F_V3F,
        gl.GL_T4F_C4F_N3F_V4F: gl.GL_T4F_C4F_N3F_V4F,
        'T4F_C4F_N3F_V4F': gl.GL_T4F_C4F_N3F_V4F,
        't4f_c4f_n3f_v4f': gl.GL_T4F_C4F_N3F_V4F,
        }
    dataFormatToDTypeMap = {
        gl.GL_V2F: gldtype('v:2f'),
        gl.GL_V3F: gldtype('v:3f'),
        gl.GL_C4UB_V2F: gldtype('c:4B, v:2f'),
        gl.GL_C4UB_V3F: gldtype('c:4B, v:3f'),
        gl.GL_C3F_V3F: gldtype('c:3f, v:3f'),
        gl.GL_N3F_V3F: gldtype('n:3f, v:3f'),
        gl.GL_C4F_N3F_V3F: gldtype('c:4f, n:3f, v:3f'),
        gl.GL_T2F_V3F: gldtype('t:2f, v:3f'),
        gl.GL_T4F_V4F: gldtype('t:4f, v:4f'),
        gl.GL_T2F_C4UB_V3F: gldtype('t:2f, c:4B, v:3f'),
        gl.GL_T2F_C3F_V3F: gldtype('t:2f, c:3f, v:3f'),
        gl.GL_T2F_N3F_V3F: gldtype('t:2f, n:3f, v:3f'),
        gl.GL_T2F_C4F_N3F_V3F: gldtype('t:2f, c:4f, n:3f, v:3f'),
        gl.GL_T4F_C4F_N3F_V4F: gldtype('t:4f, c:4f, n:3f, v:4f'),
        }

    glInterleavedArrays = staticmethod(gl.glInterleavedArrays)

    def __new__(klass, data=[], dataFormat=None, dtype=None, copy=False):
        if dtype is None:
            dataFormat = klass.dataFormatMap[dataFormat]
            dtype = klass.dataFormatToDTypeMap[dataFormat]

        data = array(data, dtype, copy=copy, order='C', ndmin=1)
        return data.view(klass)

    def __init__(self, data=[], dataFormat=None, dtype=None, copy=False):
        self._config()

    def _inferDataFormat(self):
        # we don't need to infer it because it is specified at creation time
        pass

    def enable(self):
        pass

    def bind(self):
        self.glInterleavedArrays(self.dataFormat, 0, self)

    def disable(self):
        pass

    def unbind(self):
        self.glInterleavedArrays(self.dataFormat, 0, None)


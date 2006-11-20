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

import numpy
from numpy import ndarray

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gldtype(gltypestr):
    parts = [[i.strip() for i in p.split(':', 1)] for p in gltypestr.split(',')]
    names = [e[0] for e in parts]
    formats = [e[1] for e in parts]
    return numpy.dtype(dict(names=names, formats=formats))

class NDArrayBase(ndarray):
    __array_priority__ = 25.0

    dataFormatMap = {}
    dataFormatToDTypeMap = {}
    dataFormatFromDTypeMap = {}

    def __new__(klass, shape=None, dtype=float, dataFormat=None, buffer=None, offset=0, strides=None, order=None):
        if dataFormat is not None:
            dtype, dataFormat, shape = klass.lookupDTypeFromFormat(dataFormat, shape)
        return ndarray.__new__(klass, shape, dtype, buffer, offset, strides, order)

    def __init__(self, shape=None, dtype=float, dataFormat=None, buffer=None, offset=0, strides=None, order=None):
        self._config(dataFormat)

    def _config(self, dataFormat=None):
        if dataFormat is not None:
            self.setDataFormat(dataFormat)
        elif self.dataFormat is None:
            self.inferDataFormat()

    def __array_finalize__(self, parent):
        if parent is not None:
            self._configFromParent(parent)

    def _configFromParent(self, parent):
        self.setDataFormat(parent.getDataFormat())

    @classmethod
    def lookupDTypeFromFormat(klass, dataFormat, shape):
        if isinstance(dataFormat, str):
            dataFormat = dataFormat.replace(' ', '').replace(',', '_')
            dataFormat = klass.dataFormatMap[dataFormat]

        if isinstance(shape, (int, long, float)):
            shape = (shape,)

        key = (dataFormat, shape[-1])
        dtype = klass.dataFormatToDTypeMap.get(key, None)
        if dtype is not None:
            return dtype, dataFormat, shape[:-1]

        dtype = klass.dataFormatToDTypeMap[dataFormat]
        return dtype, dataFormat, shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromFormat(klass, shape, dataFormat):
        self = klass(shape, dtype=None, dataFormat=dataFormat)
        return self

    @classmethod
    def fromData(klass, data, dtype=None, dataFormat=None, copy=False):
        if isinstance(data, klass):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2

            if (dtype2 == dtype) and (not copy):
                return data
            else:
                return data.astype(dtype)

        elif isinstance(data, ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = numpy.dtype(dtype)

            self = data.view(klass)
            if intype != data.dtype:
                self = self.astype(intype)
            elif copy: 
                self = self.copy()
            self._config(dataFormat)
            return self

        else:
            arr = numpy.array(data, dtype=dtype, copy=copy)
            self = klass(arr.shape, arr.dtype, dataFormat=dataFormat, buffer=arr)
            return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dataFormat = None
    def getDataFormat(self):
        return self.dataFormat 
    def setDataFormat(self, dataFormat):
        if isinstance(dataFormat, str):
            dataFormat = self.dataFormatMap[dataFormat]
        self.dataFormat = dataFormat

    def inferDataFormat(self, bSet=True):
        dataFormat = self.dataFormatFromDTypeMap[self.dtype.name]
        if bSet:
            self.dataFormat = dataFormat
        return dataFormat

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBase(NDArrayBase):
    drawMode = gl.GL_POINTS
    dataFormat = None
    dataFormatMap = {
        'ub': gl.GL_UNSIGNED_BYTE,
        'ui8': gl.GL_SHORT,
        'uint8': gl.GL_SHORT,
        'ubyte': gl.GL_UNSIGNED_BYTE,

        'us': gl.GL_UNSIGNED_SHORT,
        'ui16': gl.GL_SHORT,
        'uint16': gl.GL_SHORT,
        'ushort': gl.GL_UNSIGNED_SHORT,

        'b': gl.GL_BYTE,
        'i8': gl.GL_SHORT,
        'int8': gl.GL_SHORT,
        'byte': gl.GL_BYTE,

        's': gl.GL_SHORT,
        'i16': gl.GL_SHORT,
        'int16': gl.GL_SHORT,
        'short': gl.GL_SHORT,

        'i': gl.GL_INT,
        'i32': gl.GL_INT,
        'int': gl.GL_INT,
        'int32': gl.GL_INT,
        'integer': gl.GL_INT,

        'f': gl.GL_FLOAT,
        'f4': gl.GL_FLOAT,
        'float': gl.GL_FLOAT,
        'float32': gl.GL_FLOAT,

        'f8': gl.GL_DOUBLE,
        'float64': gl.GL_DOUBLE,
        'd': gl.GL_DOUBLE,
        'double': gl.GL_DOUBLE,
        }

    dataFormatDTypeMapping = [
    #    ((gl.GL_UNSIGNED_BYTE,), numpy.dtype(numpy.uint8)),
    #    ((gl.GL_UNSIGNED_SHORT,), numpy.dtype(numpy.uint16)),
    #    ((gl.GL_UNSIGNED_INT,), numpy.dtype(numpy.uint32)),

    #    ((gl.GL_BYTE,), numpy.dtype(numpy.int8)),
    #    ((gl.GL_SHORT,), numpy.dtype(numpy.int16)),
    #    ((gl.GL_INT,), numpy.dtype(numpy.int32)),

    #    ((gl.GL_FLOAT,), numpy.dtype(numpy.float32)),
    #    ((gl.GL_DOUBLE,), numpy.dtype(numpy.float64)),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    def _configFromParent(self, parent):
        NDArrayBase._configFromParent(self, parent)
        self.drawMode = parent.drawMode

    def select(self, vboOffset=None):
        self.enable()
        self.bind(vboOffset)

    def enable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def glArrayPointer(self, count, dataFormat, stride, ptr):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def bind(self, vboOffset=None):
        perElem = self.dtype[0].shape[-1]
        if vboOffset is None:
            self.glArrayPointer(perElem, self.dataFormat, self.strides[-1], self.ctypes)
        else:
            self.glArrayPointer(perElem, self.dataFormat, self.strides[-1], vboOffset)

    def deselect(self):
        self.unbind()
        self.disable()

    def disable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def unbind(self):
        pass
        #self.glArrayPointer(0, self.dataFormat, 0, None)

    glDrawArrays = staticmethod(gl.glDrawArrays)
    def draw(self, drawMode=None, vboOffset=None):
        self.select(vboOffset)
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)
    def drawRaw(self, drawMode=None):
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('v:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('v:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('v:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('v:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('v:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('v:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('v:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('v:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('v:4I')),

        ((gl.GL_BYTE, 2), gldtype('v:2b')),
        ((gl.GL_BYTE, 3), gldtype('v:3b')),
        ((gl.GL_BYTE, 4), gldtype('v:4b')),

        ((gl.GL_SHORT, 2), gldtype('v:2h')),
        ((gl.GL_SHORT, 3), gldtype('v:3h')),
        ((gl.GL_SHORT, 4), gldtype('v:4h')),

        ((gl.GL_INT, 2), gldtype('v:2i')),
        ((gl.GL_INT, 3), gldtype('v:3i')),
        ((gl.GL_INT, 4), gldtype('v:4i')),

        ((gl.GL_FLOAT, 2), gldtype('v:2f')),
        ((gl.GL_FLOAT, 3), gldtype('v:3f')),
        ((gl.GL_FLOAT, 4), gldtype('v:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('v:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('v:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('v:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_VERTEX_ARRAY
    glArrayPointer = staticmethod(gl.glVertexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class TexureCoordArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('t:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('t:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('t:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('t:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('t:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('t:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('t:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('t:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('t:4I')),

        ((gl.GL_BYTE, 2), gldtype('t:2b')),
        ((gl.GL_BYTE, 3), gldtype('t:3b')),
        ((gl.GL_BYTE, 4), gldtype('t:4b')),

        ((gl.GL_SHORT, 2), gldtype('t:2h')),
        ((gl.GL_SHORT, 3), gldtype('t:3h')),
        ((gl.GL_SHORT, 4), gldtype('t:4h')),

        ((gl.GL_INT, 2), gldtype('t:2i')),
        ((gl.GL_INT, 3), gldtype('t:3i')),
        ((gl.GL_INT, 4), gldtype('t:4i')),

        ((gl.GL_FLOAT, 2), gldtype('t:2f')),
        ((gl.GL_FLOAT, 3), gldtype('t:3f')),
        ((gl.GL_FLOAT, 4), gldtype('t:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('t:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('t:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('t:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexureCoordArray(TexureCoordArray):
    glClientActiveTexture = staticmethod(gl.glClientActiveTexture)
    texUnit = gl.GL_TEXTURE0

    def bind(self):
        self.glClientActiveTexture(self.texUnit)
        if vboOffset is None:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, self.ctypes)
        else:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, vboOffset)
    def unbind(self):
        self.glClientActiveTexture(self.texUnit)
        self.glArrayPointer(3, self.dataFormat, 0, None)

class NormalArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('n:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('n:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('n:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('n:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('n:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('n:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('n:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('n:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('n:4I')),

        ((gl.GL_BYTE, 2), gldtype('n:2b')),
        ((gl.GL_BYTE, 3), gldtype('n:3b')),
        ((gl.GL_BYTE, 4), gldtype('n:4b')),

        ((gl.GL_SHORT, 2), gldtype('n:2h')),
        ((gl.GL_SHORT, 3), gldtype('n:3h')),
        ((gl.GL_SHORT, 4), gldtype('n:4h')),

        ((gl.GL_INT, 2), gldtype('n:2i')),
        ((gl.GL_INT, 3), gldtype('n:3i')),
        ((gl.GL_INT, 4), gldtype('n:4i')),

        ((gl.GL_FLOAT, 2), gldtype('n:2f')),
        ((gl.GL_FLOAT, 3), gldtype('n:3f')),
        ((gl.GL_FLOAT, 4), gldtype('n:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('n:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('n:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('n:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_NORMAL_ARRAY
    glArrayPointer = staticmethod(gl.glNormalPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('c:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('c:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('c:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('c:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('c:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('c:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('c:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('c:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('c:4I')),

        ((gl.GL_BYTE, 2), gldtype('c:2b')),
        ((gl.GL_BYTE, 3), gldtype('c:3b')),
        ((gl.GL_BYTE, 4), gldtype('c:4b')),

        ((gl.GL_SHORT, 2), gldtype('c:2h')),
        ((gl.GL_SHORT, 3), gldtype('c:3h')),
        ((gl.GL_SHORT, 4), gldtype('c:4h')),

        ((gl.GL_INT, 2), gldtype('c:2i')),
        ((gl.GL_INT, 3), gldtype('c:3i')),
        ((gl.GL_INT, 4), gldtype('c:4i')),

        ((gl.GL_FLOAT, 2), gldtype('c:2f')),
        ((gl.GL_FLOAT, 3), gldtype('c:3f')),
        ((gl.GL_FLOAT, 4), gldtype('c:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('c:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('c:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('c:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class SecondaryColorArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('sc:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('sc:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('sc:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('sc:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('sc:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('sc:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('sc:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('sc:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('sc:4I')),

        ((gl.GL_BYTE, 2), gldtype('sc:2b')),
        ((gl.GL_BYTE, 3), gldtype('sc:3b')),
        ((gl.GL_BYTE, 4), gldtype('sc:4b')),

        ((gl.GL_SHORT, 2), gldtype('sc:2h')),
        ((gl.GL_SHORT, 3), gldtype('sc:3h')),
        ((gl.GL_SHORT, 4), gldtype('sc:4h')),

        ((gl.GL_INT, 2), gldtype('sc:2i')),
        ((gl.GL_INT, 3), gldtype('sc:3i')),
        ((gl.GL_INT, 4), gldtype('sc:4i')),

        ((gl.GL_FLOAT, 2), gldtype('sc:2f')),
        ((gl.GL_FLOAT, 3), gldtype('sc:3f')),
        ((gl.GL_FLOAT, 4), gldtype('sc:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('sc:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('sc:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('sc:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_SECONDARY_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glSecondaryColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorIndexArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('ci:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('ci:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('ci:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('ci:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('ci:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('ci:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('ci:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('ci:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('ci:4I')),

        ((gl.GL_BYTE, 2), gldtype('ci:2b')),
        ((gl.GL_BYTE, 3), gldtype('ci:3b')),
        ((gl.GL_BYTE, 4), gldtype('ci:4b')),

        ((gl.GL_SHORT, 2), gldtype('ci:2h')),
        ((gl.GL_SHORT, 3), gldtype('ci:3h')),
        ((gl.GL_SHORT, 4), gldtype('ci:4h')),

        ((gl.GL_INT, 2), gldtype('ci:2i')),
        ((gl.GL_INT, 3), gldtype('ci:3i')),
        ((gl.GL_INT, 4), gldtype('ci:4i')),

        ((gl.GL_FLOAT, 2), gldtype('ci:2f')),
        ((gl.GL_FLOAT, 3), gldtype('ci:3f')),
        ((gl.GL_FLOAT, 4), gldtype('ci:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('ci:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('ci:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('ci:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_INDEX_ARRAY
    glArrayPointer = staticmethod(gl.glIndexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class FogCoordArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('fog:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('fog:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('fog:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('fog:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('fog:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('fog:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('fog:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('fog:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('fog:4I')),

        ((gl.GL_BYTE, 2), gldtype('fog:2b')),
        ((gl.GL_BYTE, 3), gldtype('fog:3b')),
        ((gl.GL_BYTE, 4), gldtype('fog:4b')),

        ((gl.GL_SHORT, 2), gldtype('fog:2h')),
        ((gl.GL_SHORT, 3), gldtype('fog:3h')),
        ((gl.GL_SHORT, 4), gldtype('fog:4h')),

        ((gl.GL_INT, 2), gldtype('fog:2i')),
        ((gl.GL_INT, 3), gldtype('fog:3i')),
        ((gl.GL_INT, 4), gldtype('fog:4i')),

        ((gl.GL_FLOAT, 2), gldtype('fog:2f')),
        ((gl.GL_FLOAT, 3), gldtype('fog:3f')),
        ((gl.GL_FLOAT, 4), gldtype('fog:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('fog:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('fog:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('fog:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_FOG_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glFogCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class EdgeFlagArray(ArrayBase):
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 2), gldtype('edge:2B')),
        ((gl.GL_UNSIGNED_BYTE, 3), gldtype('edge:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), gldtype('edge:4B')),

        ((gl.GL_UNSIGNED_SHORT, 2), gldtype('edge:2H')),
        ((gl.GL_UNSIGNED_SHORT, 3), gldtype('edge:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), gldtype('edge:4H')),

        ((gl.GL_UNSIGNED_INT, 2), gldtype('edge:2I')),
        ((gl.GL_UNSIGNED_INT, 3), gldtype('edge:3I')),
        ((gl.GL_UNSIGNED_INT, 4), gldtype('edge:4I')),

        ((gl.GL_BYTE, 2), gldtype('edge:2b')),
        ((gl.GL_BYTE, 3), gldtype('edge:3b')),
        ((gl.GL_BYTE, 4), gldtype('edge:4b')),

        ((gl.GL_SHORT, 2), gldtype('edge:2h')),
        ((gl.GL_SHORT, 3), gldtype('edge:3h')),
        ((gl.GL_SHORT, 4), gldtype('edge:4h')),

        ((gl.GL_INT, 2), gldtype('edge:2i')),
        ((gl.GL_INT, 3), gldtype('edge:3i')),
        ((gl.GL_INT, 4), gldtype('edge:4i')),

        ((gl.GL_FLOAT, 2), gldtype('edge:2f')),
        ((gl.GL_FLOAT, 3), gldtype('edge:3f')),
        ((gl.GL_FLOAT, 4), gldtype('edge:4f')),

        ((gl.GL_DOUBLE, 2), gldtype('edge:2d')),
        ((gl.GL_DOUBLE, 3), gldtype('edge:3d')),
        ((gl.GL_DOUBLE, 4), gldtype('edge:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    glArrayType = gl.GL_EDGE_FLAG_ARRAY
    glArrayPointer = staticmethod(gl.glEdgeFlagPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


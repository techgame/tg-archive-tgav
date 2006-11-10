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

class NDArrayBase(ndarray):
    __array_priority__ = 25.0

    dataFormatMap = {}
    dataFormatToDTypeMap = {}
    dataFormatFromDTypeMap = {}

    def __new__(klass, shape=None, dtype=float, dataFormat=None, buffer=None, offset=0, strides=None, order=None):
        if dataFormat is not None:
            dtype, dataFormat = klass.lookupDTypeFromFormat(dataFormat)
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
    def lookupDTypeFromFormat(klass, dataFormat):
        if isinstance(dataFormat, str):
            dataFormat = dataFormat.replace(' ', '').replace(',', '_')
            dataFormat = klass.dataFormatMap[dataFormat]
        dtype = klass.dataFormatToDTypeMap[dataFormat]
        return dtype, dataFormat

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromFormat(klass, shape, dataFormat):
        dtype, dataFormat = klass.lookupDTypeFromFormat(dataFormat)
        self = klass(shape, dtype, dataFormat=dataFormat)
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
        (numpy.dtype(numpy.uint8), gl.GL_UNSIGNED_BYTE),
        (numpy.dtype(numpy.uint16), gl.GL_UNSIGNED_SHORT),
        (numpy.dtype(numpy.uint32), gl.GL_UNSIGNED_INT),

        (numpy.dtype(numpy.int8), gl.GL_BYTE),
        (numpy.dtype(numpy.int16), gl.GL_SHORT),
        (numpy.dtype(numpy.int32), gl.GL_INT),

        (numpy.dtype(numpy.float32), gl.GL_FLOAT),
        (numpy.dtype(numpy.float64), gl.GL_DOUBLE),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for k,v in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for k,v in dataFormatDTypeMapping)

    def select(self):
        self.enable()
        self.bind()

    def enable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def glArrayPointer(self, count, dataFormat, stride, ptr):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def bind(self):
        self.glArrayPointer(len(self), self.dataFormat, 0, self.ctypes)

    def deselect(self):
        self.unbind()
        self.disable()

    def disable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def unbind(self):
        self.glArrayPointer(0, self.dataFormat, 0, None)

    glDrawArrays = staticmethod(gl.glDrawArrays)
    def draw(self, mode):
        self.select()
        self.glDrawArrays(mode, 0, len(self.flat))
        self.deselect()

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


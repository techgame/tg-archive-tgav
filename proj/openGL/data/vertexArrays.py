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

def dtypefmt(gltypestr):
    names, formats = zip(*[[i.strip() for i in p.split(':', 1)] for p in gltypestr.split(',')])
    return numpy.dtype(dict(names=names, formats=formats))

class NDArrayBase(ndarray):
    __array_priority__ = 25.0

    dataFormatMap = {}
    dataFormatDefault = None
    defaultElementShape = (0,)

    dataFormatDTypeMapping = []
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    def __new__(klass, data=None, dataFormat=None, dtype=None, copy=False):
        return klass.fromData(data, dataFormat=dataFormat, dtype=dtype, copy=copy)

    def __init__(self, data=None, dataFormat=None, dtype=None, copy=False):
        pass

    def __array_finalize__(self, parent):
        if parent is not None:
            self._configFromParent(parent)

    def _configFromParent(self, parent):
        self.setDataFormat(parent.getDataFormat())

    dtypefmt = staticmethod(dtypefmt)

    @classmethod
    def lookupDTypeFromFormat(klass, dtype, dataFormat, shape):
        if isinstance(dataFormat, str):
            dataFormat = dataFormat.replace(' ', '').replace(',', '_')
            dataFormat = klass.dataFormatMap[dataFormat]
        else:
            dataFormat = dataFormat or klass.dataFormatDefault

        if isinstance(shape, (int, long, float)):
            shape = (shape,)
        if not shape or not shape[-1]:
            shape = shape[:-1] + klass.defaultElementShape

        dataFormatToDTypeMap = klass.dataFormatToDTypeMap
        key = (dataFormat, shape[-1])
        dtype = dataFormatToDTypeMap.get(key, None)
        if dtype is not None:
            return dtype, dataFormat, shape[:-1]

        dtype = dataFormatToDTypeMap.get(dataFormat, None)
        if dtype is not None:
            return dtype, dataFormat, shape

        raise LookupError("Not able to find an appropriate data type from format: %r, shape: %r" % (dataFormat, shape))

    @classmethod
    def lookupDefaultFormat(klass, shape):
        return klass.lookupDTypeFromFormat(None, klass.dataFormatDefault, shape)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def _fromNDArray(klass, shape=None, dtype=None, buffer=None, offset=0, strides=None, order=None, dataFormat=None):
        dtype, dataFormat, shape = klass.lookupDTypeFromFormat(dtype, dataFormat, shape)
        self = ndarray.__new__(klass, shape, dtype, buffer, offset, strides, order)
        self.setDataFormat(dataFormat)
        return self

    @classmethod
    def fromFormat(klass, shape, dataFormat=None, dtype=None):
        return klass._fromNDArray(shape, dataFormat=dataFormat)

    @classmethod
    def fromCount(klass, count, dataFormat=None, dtype=None, zeros=True, elementShape=None):
        if elementShape is None:
            elementShape = klass.defaultElementShape
        elif not isinstance(elementShape , tuple):
            elementShape = (elementShape,)

        if isinstance(count, tuple):
            shape = count + elementShape
        else: shape = (count,) + elementShape

        if zeros:
            return klass.fromZeros(shape, dataFormat=dataFormat, dtype=dtype)
        else:
            return klass.fromFormat(shape, dataFormat=dataFormat, dtype=dtype)

    @classmethod
    def fromZeros(klass, shape=None, dataFormat=None, dtype=None):
        dtype, dataFormat, shape = klass.lookupDTypeFromFormat(dtype, dataFormat, shape)
        arr = numpy.zeros(shape, dtype=dtype)
        self = ndarray.__new__(klass, shape, dtype=dtype, buffer=arr)
        self.setDataFormat(dataFormat)
        return self

    @classmethod
    def fromDataRaw(klass, data, dataFormat=None, dtype=None, copy=False):
        dtype, dataFormat, shape = klass.lookupDTypeFromFormat(dtype, dataFormat, numpy.shape(data))
        if len(dtype) == 1:
            arr = numpy.array(data, dtype=dtype[0], copy=copy)
        elif len(dtype) == 0:
            arr = numpy.array(data, dtype=dtype, copy=copy)
        else:
            raise TypeError("Unable to transform data to array type")

        self = ndarray.__new__(klass, shape, dtype=dtype, buffer=arr)
        self.setDataFormat(dataFormat)
        return self

    @classmethod
    def fromData(klass, data, dataFormat=None, dtype=None, copy=False):
        if isinstance(data, klass):
            dtype2 = data.dtype
            if dtype is None:
                self = (data if copy else data.copy())
            elif copy or dtype2 != dtype:
                self = data.astype(dtype)
            else: self = data

            self.setDataFormat(dataFormat)
            return self

        elif isinstance(data, ndarray):
            if dtype is None:
                intype = data.dtype
            else: intype = numpy.dtype(dtype)

            self = data.view(klass)
            if intype != data.dtype:
                self = self.astype(intype)
            elif copy: 
                self = self.copy()

            self.setDataFormat(dataFormat)
            return self

        elif data is None:
            return klass.fromZeros(None, dataFormat=dataFormat, dtype=dtype)

        else:
            return klass.fromDataRaw(data, dataFormat, dtype, copy)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dataFormat = None
    def getDataFormat(self):
        return self.dataFormat 
    def setDataFormat(self, dataFormat):
        if isinstance(dataFormat, str):
            dataFormat = self.dataFormatMap[dataFormat]
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

    dataFormatToImmediateFn = {}

    def _configFromParent(self, parent):
        NDArrayBase._configFromParent(self, parent)
        self.drawMode = parent.drawMode
        self._glImmediate_ = parent._glImmediate_

    def _config(self, dataFormat):
        NDArrayBase._config(self, dataFormat)
        self._configImmediateFn()

    def select(self, vboOffset=None):
        self.enable()
        self.bind(vboOffset)

    def enable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def glArrayPointer(self, count, dataFormat, stride, ptr):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def bind(self, vboOffset=None):
        elemSize = self.dtype[0].shape[-1]
        if vboOffset is None:
            self.glArrayPointer(elemSize, self.dataFormat, self.strides[-1], self.ctypes)
        else:
            self.glArrayPointer(elemSize, self.dataFormat, self.strides[-1], vboOffset)

    def deselect(self):
        self.unbind()
        self.disable()

    def disable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def unbind(self):
        pass

    glDrawArrays = staticmethod(gl.glDrawArrays)
    def draw(self, drawMode=None, vboOffset=None):
        self.select(vboOffset)
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)
    def drawRaw(self, drawMode=None):
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def glImmediate(self):
        self._glImmediate_()
    def _configImmediateFn(self):
        key = (self.dataFormat)#, self.dtype[0].shape[-1])
        glImmediateFn = self.dataFormatToImmediateFn[key]
        self._glImmediate_ = partial(glImmediateFn, self.ctypes.data_as(glImmediateFn.api.argtypes[-1]))
    def _glImmediate_(self, ptr):
        raise NotImplementedError('_glImmediate_ should be repopulated from the dataformat and shape of the array')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (3,)
    dataFormatDTypeMapping = [
        ((gl.GL_SHORT, 2), dtypefmt('v:2h')),
        ((gl.GL_SHORT, 3), dtypefmt('v:3h')),
        ((gl.GL_SHORT, 4), dtypefmt('v:4h')),

        ((gl.GL_INT, 2), dtypefmt('v:2i')),
        ((gl.GL_INT, 3), dtypefmt('v:3i')),
        ((gl.GL_INT, 4), dtypefmt('v:4i')),

        ((gl.GL_FLOAT, 2), dtypefmt('v:2f')),
        ((gl.GL_FLOAT, 3), dtypefmt('v:3f')),
        ((gl.GL_FLOAT, 4), dtypefmt('v:4f')),

        ((gl.GL_DOUBLE, 2), dtypefmt('v:2d')),
        ((gl.GL_DOUBLE, 3), dtypefmt('v:3d')),
        ((gl.GL_DOUBLE, 4), dtypefmt('v:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_SHORT, 2), gl.glVertex2sv),
        ((gl.GL_SHORT, 3), gl.glVertex3sv),
        ((gl.GL_SHORT, 4), gl.glVertex4sv),

        ((gl.GL_INT, 2), gl.glVertex2iv),
        ((gl.GL_INT, 3), gl.glVertex3iv),
        ((gl.GL_INT, 4), gl.glVertex4iv),

        ((gl.GL_FLOAT, 2), gl.glVertex2fv),
        ((gl.GL_FLOAT, 3), gl.glVertex3fv),
        ((gl.GL_FLOAT, 4), gl.glVertex4fv),

        ((gl.GL_DOUBLE, 2), gl.glVertex2dv),
        ((gl.GL_DOUBLE, 3), gl.glVertex3dv),
        ((gl.GL_DOUBLE, 4), gl.glVertex4dv),
        ])

    glArrayType = gl.GL_VERTEX_ARRAY
    glArrayPointer = staticmethod(gl.glVertexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class TexureCoordArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (3,)
    dataFormatDTypeMapping = [
        ((gl.GL_SHORT, 1), dtypefmt('t:1h')),
        ((gl.GL_SHORT, 2), dtypefmt('t:2h')),
        ((gl.GL_SHORT, 3), dtypefmt('t:3h')),
        ((gl.GL_SHORT, 4), dtypefmt('t:4h')),

        ((gl.GL_INT, 1), dtypefmt('t:1i')),
        ((gl.GL_INT, 2), dtypefmt('t:2i')),
        ((gl.GL_INT, 3), dtypefmt('t:3i')),
        ((gl.GL_INT, 4), dtypefmt('t:4i')),

        ((gl.GL_FLOAT, 1), dtypefmt('t:1f')),
        ((gl.GL_FLOAT, 2), dtypefmt('t:2f')),
        ((gl.GL_FLOAT, 3), dtypefmt('t:3f')),
        ((gl.GL_FLOAT, 4), dtypefmt('t:4f')),

        ((gl.GL_DOUBLE, 1), dtypefmt('t:1d')),
        ((gl.GL_DOUBLE, 2), dtypefmt('t:2d')),
        ((gl.GL_DOUBLE, 3), dtypefmt('t:3d')),
        ((gl.GL_DOUBLE, 4), dtypefmt('t:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_SHORT, 1), gl.glTexCoord1sv),
        ((gl.GL_SHORT, 2), gl.glTexCoord2sv),
        ((gl.GL_SHORT, 3), gl.glTexCoord3sv),
        ((gl.GL_SHORT, 4), gl.glTexCoord4sv),

        ((gl.GL_INT, 1), gl.glTexCoord1iv),
        ((gl.GL_INT, 2), gl.glTexCoord2iv),
        ((gl.GL_INT, 3), gl.glTexCoord3iv),
        ((gl.GL_INT, 4), gl.glTexCoord4iv),

        ((gl.GL_FLOAT, 1), gl.glTexCoord1fv),
        ((gl.GL_FLOAT, 2), gl.glTexCoord2fv),
        ((gl.GL_FLOAT, 3), gl.glTexCoord3fv),
        ((gl.GL_FLOAT, 4), gl.glTexCoord4fv),

        ((gl.GL_DOUBLE, 1), gl.glTexCoord1dv),
        ((gl.GL_DOUBLE, 2), gl.glTexCoord2dv),
        ((gl.GL_DOUBLE, 3), gl.glTexCoord3dv),
        ((gl.GL_DOUBLE, 4), gl.glTexCoord4dv),
        ])

    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexureCoordArray(TexureCoordArray):
    glClientActiveTexture = staticmethod(gl.glClientActiveTexture)
    texUnit = gl.GL_TEXTURE0

    dataFormatToImmediateFn = [
        ((gl.GL_SHORT, 1), gl.glMultiTexCoord1sv),
        ((gl.GL_SHORT, 2), gl.glMultiTexCoord2sv),
        ((gl.GL_SHORT, 3), gl.glMultiTexCoord3sv),
        ((gl.GL_SHORT, 4), gl.glMultiTexCoord4sv),

        ((gl.GL_INT, 1), gl.glMultiTexCoord1iv),
        ((gl.GL_INT, 2), gl.glMultiTexCoord2iv),
        ((gl.GL_INT, 3), gl.glMultiTexCoord3iv),
        ((gl.GL_INT, 4), gl.glMultiTexCoord4iv),

        ((gl.GL_FLOAT, 1), gl.glMultiTexCoord1fv),
        ((gl.GL_FLOAT, 2), gl.glMultiTexCoord2fv),
        ((gl.GL_FLOAT, 3), gl.glMultiTexCoord3fv),
        ((gl.GL_FLOAT, 4), gl.glMultiTexCoord4fv),

        ((gl.GL_DOUBLE, 1), gl.glMultiTexCoord1dv),
        ((gl.GL_DOUBLE, 2), gl.glMultiTexCoord2dv),
        ((gl.GL_DOUBLE, 3), gl.glMultiTexCoord3dv),
        ((gl.GL_DOUBLE, 4), gl.glMultiTexCoord4dv),
        ]

    def bind(self):
        self.glClientActiveTexture(self.texUnit)
        if vboOffset is None:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, self.ctypes)
        else:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, vboOffset)
    def unbind(self):
        self.glClientActiveTexture(self.texUnit)
        self.glArrayPointer(3, self.dataFormat, 0, None)
    
    def glImmediate(self):
        self._glImmediate_(self.texUnit, self.ctypes)

class NormalArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (3,)
    dataFormatDTypeMapping = [
        ((gl.GL_BYTE, 3), dtypefmt('n:3b')),
        ((gl.GL_SHORT, 3), dtypefmt('n:3h')),
        ((gl.GL_INT, 3), dtypefmt('n:3i')),
        ((gl.GL_FLOAT, 3), dtypefmt('n:3f')),
        ((gl.GL_DOUBLE, 3), dtypefmt('n:3d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_BYTE, 3), gl.glNormal3bv),
        ((gl.GL_SHORT, 3), gl.glNormal3sv),
        ((gl.GL_INT, 3), gl.glNormal3iv),
        ((gl.GL_FLOAT, 3), gl.glNormal3fv),
        ((gl.GL_DOUBLE, 3), gl.glNormal3dv),
        ])

    glArrayType = gl.GL_NORMAL_ARRAY
    glArrayPointer = staticmethod(gl.glNormalPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (4,)
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 3), dtypefmt('c:3B')),
        ((gl.GL_UNSIGNED_BYTE, 4), dtypefmt('c:4B')),

        ((gl.GL_UNSIGNED_SHORT, 3), dtypefmt('c:3H')),
        ((gl.GL_UNSIGNED_SHORT, 4), dtypefmt('c:4H')),

        ((gl.GL_UNSIGNED_INT, 3), dtypefmt('c:3I')),
        ((gl.GL_UNSIGNED_INT, 4), dtypefmt('c:4I')),

        ((gl.GL_BYTE, 3), dtypefmt('c:3b')),
        ((gl.GL_BYTE, 4), dtypefmt('c:4b')),

        ((gl.GL_SHORT, 3), dtypefmt('c:3h')),
        ((gl.GL_SHORT, 4), dtypefmt('c:4h')),

        ((gl.GL_INT, 3), dtypefmt('c:3i')),
        ((gl.GL_INT, 4), dtypefmt('c:4i')),

        ((gl.GL_FLOAT, 3), dtypefmt('c:3f')),
        ((gl.GL_FLOAT, 4), dtypefmt('c:4f')),

        ((gl.GL_DOUBLE, 3), dtypefmt('c:3d')),
        ((gl.GL_DOUBLE, 4), dtypefmt('c:4d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_UNSIGNED_BYTE, 3), gl.glColor3ubv),
        ((gl.GL_UNSIGNED_BYTE, 4), gl.glColor4ubv),

        ((gl.GL_UNSIGNED_SHORT, 3), gl.glColor3usv),
        ((gl.GL_UNSIGNED_SHORT, 4), gl.glColor4usv),

        ((gl.GL_UNSIGNED_INT, 3), gl.glColor3uiv),
        ((gl.GL_UNSIGNED_INT, 4), gl.glColor4uiv),

        ((gl.GL_BYTE, 3), gl.glColor3bv),
        ((gl.GL_BYTE, 4), gl.glColor4bv),

        ((gl.GL_SHORT, 3), gl.glColor3sv),
        ((gl.GL_SHORT, 4), gl.glColor4sv),

        ((gl.GL_INT, 3), gl.glColor3iv),
        ((gl.GL_INT, 4), gl.glColor4iv),

        ((gl.GL_FLOAT, 3), gl.glColor3fv),
        ((gl.GL_FLOAT, 4), gl.glColor4fv),

        ((gl.GL_DOUBLE, 3), gl.glColor3dv),
        ((gl.GL_DOUBLE, 4), gl.glColor4dv),
        ])

    glArrayType = gl.GL_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class SecondaryColorArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (3,)
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 3), dtypefmt('c2:3B')),
        ((gl.GL_UNSIGNED_SHORT, 3), dtypefmt('c2:3H')),
        ((gl.GL_UNSIGNED_INT, 3), dtypefmt('c2:3I')),
        ((gl.GL_BYTE, 3), dtypefmt('c2:3b')),
        ((gl.GL_SHORT, 3), dtypefmt('c2:3h')),
        ((gl.GL_INT, 3), dtypefmt('c2:3i')),
        ((gl.GL_FLOAT, 3), dtypefmt('c2:3f')),
        ((gl.GL_DOUBLE, 3), dtypefmt('c2:3d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_UNSIGNED_BYTE, 3), gl.glSecondaryColor3ubv),
        ((gl.GL_UNSIGNED_SHORT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_UNSIGNED_INT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_BYTE, 3), gl.glSecondaryColor3usv),
        ((gl.GL_SHORT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_INT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_FLOAT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_DOUBLE, 3), gl.glSecondaryColor3usv),
        ])

    glArrayType = gl.GL_SECONDARY_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glSecondaryColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorIndexArray(ArrayBase):
    dataFormatDefault = gl.GL_UNSIGNED_BYTE
    defaultElementShape = (1,)
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 1), dtypefmt('ci:1B')),
        ((gl.GL_SHORT, 1), dtypefmt('ci:1h')),
        ((gl.GL_INT, 1), dtypefmt('ci:1i')),
        ((gl.GL_FLOAT, 1), dtypefmt('ci:1f')),
        ((gl.GL_DOUBLE, 1), dtypefmt('ci:1d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_UNSIGNED_BYTE, 1), gl.glIndexubv),
        ((gl.GL_SHORT, 1), gl.glIndexsv),
        ((gl.GL_INT, 1), gl.glIndexiv),
        ((gl.GL_FLOAT, 1), gl.glIndexfv),
        ((gl.GL_DOUBLE, 1), gl.glIndexdv),
        ])

    glArrayType = gl.GL_INDEX_ARRAY
    glArrayPointer = staticmethod(gl.glIndexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class FogCoordArray(ArrayBase):
    dataFormatDefault = gl.GL_FLOAT
    defaultElementShape = (1,)
    dataFormatDTypeMapping = [
        ((gl.GL_FLOAT, 1), dtypefmt('fog:1f')),
        ((gl.GL_DOUBLE, 1), dtypefmt('fog:1d')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_FLOAT, 1), gl.glFogCoordfv),
        ((gl.GL_DOUBLE, 1), gl.glFogCoorddv),
        ])

    glArrayType = gl.GL_FOG_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glFogCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class EdgeFlagArray(ArrayBase):
    dataFormatDefault = gl.GL_UNSIGNED_BYTE
    defaultElementShape = (1,)
    dataFormatDTypeMapping = [
        ((gl.GL_UNSIGNED_BYTE, 1), dtypefmt('edge:1B')),
        ]
    dataFormatFromDTypeMap = dict((k.name, v) for v,k in dataFormatDTypeMapping)
    dataFormatToDTypeMap = dict((v, k) for v,k in dataFormatDTypeMapping)

    dataFormatToImmediateFn = dict([
        ((gl.GL_UNSIGNED_BYTE, 1), gl.glEdgeFlagv),
        ])

    glArrayType = gl.GL_EDGE_FLAG_ARRAY
    glArrayPointer = staticmethod(gl.glEdgeFlagPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


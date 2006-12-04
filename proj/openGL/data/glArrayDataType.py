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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayDataType(object):
    defaultFormat = None

    dtypeMap = {}

    gltypeidMap = {
        'uint8': gl.GL_UNSIGNED_BYTE,
        'B': gl.GL_UNSIGNED_BYTE,
        'int8': gl.GL_BYTE,
        'b': gl.GL_BYTE,
        'uint16': gl.GL_UNSIGNED_SHORT,
        'H': gl.GL_UNSIGNED_SHORT,
        'int16': gl.GL_SHORT,
        'h': gl.GL_SHORT,
        'uint32': gl.GL_UNSIGNED_INT,
        'I': gl.GL_UNSIGNED_INT,
        'L': gl.GL_UNSIGNED_INT,
        'int32': gl.GL_INT,
        'i': gl.GL_INT,
        'l': gl.GL_INT,
        'float32': gl.GL_FLOAT,
        'f': gl.GL_FLOAT,
        'float64': gl.GL_DOUBLE,
        'd': gl.GL_DOUBLE,
        }

    def __init__(self, other=None):
        if other is not None:
            self.copyFrom(other)
        else:
            self.dtypeMap = self.dtypeMap.copy()

    @classmethod
    def new(klass, other=None):
        return klass(other)

    def copy(self):
        return self.new(self)

    def copyFrom(self, other):
        self.defaultFormat = other.defaultFormat
        self.dtypeMap = other.dtypeMap.copy()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addFormatGroups(self, dtypesList, entrySizes=(), default=NotImplemented):
        dtypeMap = self.dtypeMap
        keyForDtype = self._getKeyForDtype
        for edtype in dtypesList:
            edtype = self.dtypefmt(edtype)

            for esize in entrySizes:
                if esize is not None:
                    dt = numpy.dtype((edtype, esize))
                else: dt = edtype
                dtypeMap[keyForDtype(dt)] = dt

        if default is not NotImplemented:
            self.setDefaultFormat(default)

    def setDefaultFormat(self, format):
        self.defaultFormat = self.dtypefmt(format)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def configFrom(self, array, parent=None):
        if parent is not None:
            array.gltypeid = parent.gltypeid 
        else: array.gltypeid = self.gltypeidForDtype(array.dtype)

    def gltypeidForArray(self, array):
        return self.gltypeidForDtype(array.dtype)
    def gltypeidForDtype(self, dtype):
        key = self._getKeyForDtype(dtype)
        return self.gltypeidMap[key]

    @classmethod
    def _getKeyForDtype(klass, dtype, shape=None):
        if shape is None:
            shape = dtype.shape

        if dtype.kind != 'V':
            if shape and sum(shape)>1:
                assert len(shape) == 1, shape
                return '%s%s' % (shape[0], dtype.char)
            else: return dtype.char
        elif dtype.base.kind != 'V':
            if shape and sum(shape)>1:
                assert len(shape) == 1, shape
                return '%s%s' % (shape[0], dtype.base.char)
            else: 
                raise RuntimeError("Unexpected dtype: %r" % (dtype,))
        else:
            keyForDtype = klass._getKeyForDtype
            return ';'.join(name+':'+keyForDtype(dtype.fields[name][0]) for name in dtype.names)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dtypeFrom = staticmethod(numpy.dtype)

    @classmethod
    def dtypefmt(klass, dtypefmt):
        if isinstance(dtypefmt, basestring):
            if (':' in dtypefmt):
                names, formats = zip(*[[i.strip() for i in p.split(':', 1)] for p in dtypefmt.split(';') if p])
                dtypefmt = dict(names=names, formats=formats)
            else:
                result = klass.dtypeMap.get(dtypefmt, None)
                if result is not None:
                    return result
        elif isinstance(dtypefmt, (int, long)):
            return klass.gltypeidToDtype[dtypefmt]

        return klass.dtypeFrom(dtypefmt)

    def lookupDTypeFrom(self, dtype, shape, completeShape=None):
        if completeShape is None:
            completeShape = isinstance(shape, tuple) and dtype is None

        print 'lookupDTypeFrom:', (dtype, shape, completeShape)
        if dtype is None:
            dtype = self.defaultFormat
        else:
            dtype = self.dtypefmt(dtype)
            key = (dtype.base.name, shape[-1:])

        if isinstance(shape, (int, long, float)):
            shape = (shape,)
        elif shape is None: shape = ()

        if completeShape:
            key = self._getKeyForDtype(dtype, shape[-1:])
            shape = shape[:-1]
        else: key = self._getKeyForDtype(dtype)

        print '   ...', (self.dtypeMap.get(key, '<no entry>'), shape, key)
        dtype = self.dtypeMap[key]
        return dtype, shape

    @classmethod
    def gltypeidToDtypePopulate(klass):
        gltypeidToDtype = {}
        keyForDtype = klass._getKeyForDtype
        for dtype, gltypeid in klass.gltypeidMap.iteritems():
            dtype = klass.dtypefmt(dtype)
            gltypeidToDtype[gltypeid] = dtype

            if dtype.kind == 'V':
                klass.dtypeMap[keyForDtype(dtype)] = dtype
            
        klass.gltypeidToDtype = gltypeidToDtype

GLArrayDataType.gltypeidToDtypePopulate()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLInterleavedArrayDataType(GLArrayDataType):
    gltypeidMap = {
        'v:2f': gl.GL_V2F,
        'v:3f': gl.GL_V3F,
        'c:4B;v:2f': gl.GL_C4UB_V2F,
        'c:4B;v:3f': gl.GL_C4UB_V3F,
        'c:3f;v:3f': gl.GL_C3F_V3F,
        'n:3f;v:3f': gl.GL_N3F_V3F,
        'c:4f;n:3f;v:3f': gl.GL_C4F_N3F_V3F,
        't:2f;v:3f': gl.GL_T2F_V3F,
        't:4f;v:4f': gl.GL_T4F_V4F,
        't:2f;c:4B;v:3f': gl.GL_T2F_C4UB_V3F,
        't:2f;c:3f;v:3f': gl.GL_T2F_C3F_V3F,
        't:2f;n:3f;v:3f': gl.GL_T2F_N3F_V3F,
        't:2f;c:4f;n:3f;v:3f': gl.GL_T2F_C4F_N3F_V3F,
        't:4f;c:4f;n:3f;v:4f': gl.GL_T4F_C4F_N3F_V4F,
        }
GLInterleavedArrayDataType.gltypeidToDtypePopulate()


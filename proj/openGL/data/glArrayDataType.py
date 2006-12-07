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
from . import glArrayInfo 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayDataType(object):
    defaultFormat = None

    dtypeMap = {}
    _glTypeIdMap = glArrayInfo.glTypeIdMap

    def __init__(self, other=None):
        if other is not None:
            self.copyFrom(other)

    @classmethod
    def new(klass, other=None):
        return klass(other)

    def copy(self):
        return self.new(self)

    def copyFrom(self, other):
        self.defaultFormat = other.defaultFormat
        self.kind = other.kind
        self.glKindId = other.glKindId

        if self.dtypeMap is not other.dtypeMap:
            self.dtypeMap = other.dtypeMap.copy()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    kind = None
    glKindId = None
    glArrayPointer = None
    def setKind(self, kind):
        self.kind = kind
        self.glKindId = glArrayInfo.glKindIdFrom(kind)
        self._glImmediateFnMap = glArrayInfo.glImmediateMapFrom(kind)
        self.glArrayPointer = glArrayInfo.glArrayPointerFnFrom(kind)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def addFormatGroups(self, dtypesList, entrySizes=(), default=NotImplemented):
        dtypeMap = self.dtypeMap.copy()
        keyForDtype = self._getKeyForDtype
        for edtype in dtypesList:
            edtype = self.dtypefmt(edtype)

            for esize in entrySizes:
                if esize is not None:
                    dt = numpy.dtype((edtype, esize))
                else: dt = edtype
                dtypeMap[keyForDtype(dt)] = dt

        self.dtypeMap = dtypeMap
        if default is not NotImplemented:
            self.setDefaultFormat(default)

    def setDefaultFormat(self, format):
        self.defaultFormat = self.dtypefmt(format)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def configFrom(self, array, parent=None):
        glTypeId = getattr(parent, 'glTypeId', None)
        if glTypeId is None:
            glTypeId = self.glTypeIdForArray(array)
        array.glTypeId = glTypeId

    def glTypeIdForArray(self, array):
        return self.glTypeIdForDtype(array.dtype)
    def glTypeIdForDtype(self, dtype):
        key = self._getKeyForDtype(dtype)
        return self._glTypeIdMap.get(key, None)

    def gldrawModeFor(self, key):
        return self.gldrawModeMap.get(key, key)

    _glImmediateFnMap = {}
    def glImmediateFor(self, array):
        fnByShape = self._glImmediateFnMap.get(array.glTypeId) or {}
        glImmediate = fnByShape.get(array.shape[-1])
        return glImmediate

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
            return klass._glTypeIdToDtype[dtypefmt]

        return klass.dtypeFrom(dtypefmt)

    def lookupDTypeFrom(self, dtype, shape, completeShape=None):
        if completeShape is None:
            completeShape = (dtype is None) and isinstance(shape, tuple)
        if completeShape and shape[-1] == -1:
            completeShape = False
            shape = shape[:-1]

        if dtype is None:
            dtype = self.defaultFormat
        else:
            dtype = self.dtypefmt(dtype)
            key = (dtype.base.name, shape[-1:])

        if isinstance(shape, (int, long)): shape = (shape,)
        elif shape is None: shape = ()

        if completeShape:
            key = self._getKeyForDtype(dtype, shape[-1:])
            shape = shape[:-1]
        else: key = self._getKeyForDtype(dtype)

        dtype = self.dtypeMap[key]
        return dtype, shape

    @classmethod
    def _glTypeIdToDtypePopulate(klass):
        glTypeIdToDtype = {}
        keyForDtype = klass._getKeyForDtype
        for dtype, glTypeId in klass._glTypeIdMap.iteritems():
            dtype = klass.dtypefmt(dtype)
            glTypeIdToDtype[glTypeId] = dtype

            if dtype.kind == 'V':
                klass.dtypeMap[keyForDtype(dtype)] = dtype
            
        klass._glTypeIdToDtype = glTypeIdToDtype

GLArrayDataType._glTypeIdToDtypePopulate()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLInterleavedArrayDataType(GLArrayDataType):
    _glTypeIdMap = glArrayInfo.glInterleavedTypeIdMap

GLInterleavedArrayDataType._glTypeIdToDtypePopulate()


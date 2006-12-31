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
from numpy import atleast_2d, ndarray

from .glDataProperty import GLDataProperty, asDataProperty
from .glArrayDataType import GLBaseArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayBase(ndarray):
    __array_priority__ = 25.0

    gldtype = GLBaseArrayDataType()
    glTypeId = None

    useDefault = object()
    default = numpy.array([0], 'B')

    def __new__(klass, data=None, dtype=None, shape=None, copy=False):
        if isinstance(data, klass):
            # fastpath applying self class to self
            if not copy and dtype is None and not shape:
                return data

        if not shape:
            if isinstance(data, (int, long)):
                copy = True
                shape = (data,) + klass.default.shape
                data = klass.default
            elif data is None:
                copy = True
                data = klass.default
            return klass.fromData(data, dtype, shape, copy)
        elif data is not None:
            return klass.fromData(data, dtype, shape, copy)
        else:
            return klass.fromShape(shape, dtype)

    def __init__(self, data=None, dtype=None, shape=None, copy=False):
        pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __nonzero__(self):
        return self.size != 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _glTypeId = None
    @property
    def glTypeId(self):
        r = self._glTypeId
        if r is None:
            r = self.gldtype.glTypeIdForArray(self)
            self._glTypeId = r 
        return r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Contruction methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromShape(klass, shape, dtype=None):
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype)
        if shape[-1:] == (-1,):
            shape = shape[:-1]
        self = ndarray.__new__(klass, shape, dtype=dtype, order=order)
        return self

    @classmethod
    def fromData(klass, data, dtype=None, shape=None, copy=False):
        if shape is not None:
            return klass.fromDataRaw(data, dtype, shape, copy)

        if isinstance(data, ndarray):
            return klass.fromDataArray(data, dtype, copy)

        return klass.fromDataRaw(data, dtype, shape, copy)

    @classmethod
    def fromDataArray(klass, data, dtype, copy=False):
        indtype = klass.gldtype.lookupDTypeFrom(dtype, data.shape, data.dtype)[0]

        if indtype.shape == data.shape[-1:]:
            indtype = indtype.base
        result = data.view(klass)
        
        if copy or indtype != result.dtype:
            result = result.astype(indtype)

        return klass._normalized(result)

    @classmethod
    def fromDataRaw(klass, data, dtype=None, shape=None, copy=False):
        data = klass._valueFrom(data, (dtype, ()))
        if (not shape) and isinstance(data, klass):
            if dtype is None or data.dtype == dtype:
                return data

        dataShape = numpy.shape(data)
        shape = shape or dataShape
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype, shape)
        if shape[-1:] == (-1,):
            shape = shape[:-1]
        elif dtype.shape == shape[-1:]:
            shape = shape[:-1]

        self = ndarray.__new__(klass, shape, dtype, order=order)
        self = klass._normalized(self)
        self.view(ndarray)[..., :] = data
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def _normalized(klass, result):
        return atleast_2d(result)

    @classmethod
    def _valueFrom(klass, value, edtype):
        if isinstance(value, ndarray):
            return value
        return numpy.asarray(value)

    def setPropValue(self, value):
        self[:] = value
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    PropertyFactory = GLDataProperty
    property = classmethod(asDataProperty)
    asProperty = asDataProperty


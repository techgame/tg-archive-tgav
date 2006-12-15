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

from .observableData import ObservableData

from .glArrayDataType import GLBaseArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayBase(ndarray, ObservableData):
    __array_priority__ = 25.0

    gldtype = GLBaseArrayDataType()
    glTypeId = None

    _atleast_nd = staticmethod(atleast_2d)

    useDefault = object()
    default = numpy.array(0, 'B')

    def __new__(klass, data=None, dtype=None, shape=None, copy=False):
        klass._visitOnObservableNew(klass)
        if not shape:
            if data is None or isinstance(data, (int, long)):
                copy = True
                if data is not None: 
                    shape = (data,) + klass.default.shape
                data = klass.default
            return klass.fromData(data, dtype, shape, copy)
        elif data is not None:
            return klass.fromData(data, dtype, shape, copy)
        else:
            return klass.fromShape(shape, dtype)

    def __init__(self, data=None, dtype=None, shape=None, copy=False):
        ObservableData.__init__(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __array_finalize__(self, parent):
        if parent is not None:
            self._configFromParent(parent)

    def _configFromParent(self, parent):
        self.gldtype.configFrom(self, parent)

    def __nonzero__(self):
        return self.size != 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Contruction methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromShape(klass, shape, dtype=None):
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype)
        if shape[-1:] == (-1,):
            shape = shape[:-1]
        self = ndarray.__new__(klass, shape, dtype=dtype, order=order)
        self.gldtype.configFrom(self)
        return klass._atleast_nd(self)

    @classmethod
    def fromData(klass, data, dtype=None, shape=None, copy=False):
        if shape is not None:
            return klass.fromDataRaw(data, dtype, shape, copy)

        elif isinstance(data, klass):
            dtype2 = data.dtype
            if dtype is None:
                self = (data.copy() if copy else data)
            elif copy or dtype2 != dtype:
                self = data.astype(dtype)
            else: self = data

            return klass._atleast_nd(self)

        elif isinstance(data, ndarray):
            intype = klass.gldtype.dtypefmt(dtype, data.dtype)

            self = data.view(klass)
            if intype != data.dtype:
                self = self.astype(intype)
            elif copy: 
                self = self.copy()

            return klass._atleast_nd(self)

        else:
            return klass.fromDataRaw(data, dtype, shape, copy)

    @classmethod
    def fromDataRaw(klass, data, dtype=None, shape=None, copy=False):
        data = klass._valueFrom(data, (dtype, ()))
        if (not shape) and isinstance(data, klass):
            if dtype is None or data.dtype == dtype:
                return data

        dataShape = numpy.shape(data)
        shape = shape or dataShape
        if shape[-1:] == (-1,):
            shape = shape[:-1]
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype, shape)
        if dtype.shape == shape[-1:]:
            shape = shape[:-1]

        self = ndarray.__new__(klass, shape, dtype, order=order)
        self.gldtype.configFrom(self)
        self = klass._atleast_nd(self)
        self.view(ndarray)[..., :] = data
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __setslice__(self, i, j, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setslice__(self, i, j, value)
        self._kvnotify_('set', 'slice', (i, j))
    def __delslice__(self, i, j):
        ndarray.__delslice__(self, i, j)
        self._kvnotify_('del', 'slice', (i, j))
    def __setitem__(self, index, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setitem__(self, index, value)
        self._kvnotify_('set', 'index', index)
    def __delitem__(self, i): 
        ndarray.__delitem__(self, i)
        self._kvnotify_('del', 'index', i)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def _valueFrom(klass, value, edtype):
        if isinstance(value, ndarray):
            return value
        return numpy.asarray(value)


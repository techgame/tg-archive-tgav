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
from .glArrayDataType import GLBaseArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayBase(ndarray):
    __array_priority__ = 25.0
    gldtype = GLBaseArrayDataType()
    glTypeId = None

    useDefault = object()
    default = numpy.array(0, 'B')

    def __new__(klass, data=None, dtype=None, shape=None, copy=False):
        if not shape:
            if data is None or isinstance(data, (int, long)):
                copy = True
                if data: 
                    shape = (data,) + klass.default.shape
                data = klass.default
            return klass.fromData(data, dtype, shape, copy)
        elif data is not None:
            return klass.fromData(data, dtype, shape, copy)
        else:
            return klass.fromShape(shape, dtype)

    def __init__(klass, data=None, dtype=None, shape=None, copy=False):
        pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __array_finalize__(self, parent):
        if parent is not None:
            self._configFromParent(parent)

    def _configFromParent(self, parent):
        self.gldtype.configFrom(self, parent)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Contruction methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromShape(klass, shape, dtype=None):
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype)
        self = ndarray.__new__(klass, shape, dtype=dtype, order=order)
        self.gldtype.configFrom(self)
        return atleast_2d(self)

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

            return atleast_2d(self)

        elif isinstance(data, ndarray):
            intype = klass.gldtype.dtypefmt(dtype, data.dtype)

            self = data.view(klass)
            if intype != data.dtype:
                self = self.astype(intype)
            elif copy: 
                self = self.copy()

            return atleast_2d(self)

        else:
            return klass.fromDataRaw(data, dtype, shape, copy)

    @classmethod
    def fromDataRaw(klass, data, dtype=None, shape=None, copy=False):
        data = klass._valueFrom(data, (dtype, ()))
        if (not shape) and isinstance(data, klass):
            if dtype is None or data.dtype == dtype:
                return data

        shape = shape or numpy.shape(data)
        dtype, order = klass.gldtype.lookupDTypeFrom(dtype, shape)
        if dtype.shape > 0:
            shape = shape[:-1] or (1,)

        self = ndarray.__new__(klass, shape, dtype, order=order)
        self.gldtype.configFrom(self)
        self = atleast_2d(self)
        self.view(ndarray)[..., :] = data
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _kvnotify_(self, op, key):
        """This method is intended to be replaced by a mixin with ObservableObject"""

    def __setslice__(self, i, j, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setslice__(self, i, j, value)
        self._kvnotify_('set', (i, j))
    def __delslice__(self, i, j):
        ndarray.__delslice__(self, i, j)
        self._kvnotify_('del', (i, j))
    def __setitem__(self, index, value):
        value = self._valueFrom(value, self.edtype)
        ndarray.__setitem__(self, index, value)
        self._kvnotify_('set', index)
    def __delitem__(self, i): 
        ndarray.__delitem__(self, i)
        self._kvnotify_('del', i)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def _valueFrom(klass, value, edtype):
        if isinstance(value, ndarray):
            return value
        return numpy.asarray(value)


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
from numpy import ndarray
from .glArrayDataType import GLArrayDataType

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLArrayBase(ndarray):
    __array_priority__ = 25.0
    gldtype = GLArrayDataType()
    gltypeid = None

    useDefault = object()
    default = numpy.array([0], 'B')

    def __new__(klass, data=None, dtype=None, shape=None, copy=False, value=useDefault):
        if data is None:
            # default with no args is to use shape a single 1-element
            self = klass.fromShape(shape or 1, dtype, value)
        elif isinstance(data, (int, long)):
            # determine if shape is complete before we adjust the shape parameter
            completeShape = (dtype is None) and isinstance(shape, tuple)
            # adjust shape
            shape = (data,) + (shape or ())
            self = klass.fromShape(shape, dtype, value, completeShape)
        else:
            self = klass.fromData(data, dtype, copy)
        return self

    def __init__(klass, data=None, dtype=None, shape=(), copy=False, value=useDefault):
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
    def fromZeros(klass, shape, dtype=None):
        return klass.fromShape(shape, dtype, value=0)

    @classmethod
    def fromOnes(klass, shape, dtype=None):
        return klass.fromShape(shape, dtype, value=1)

    @classmethod
    def fromShape(klass, shape, dtype=None, value=None, completeShape=None):
        dtype, shape = klass.gldtype.lookupDTypeFrom(dtype, shape, completeShape)
        self = ndarray.__new__(klass, shape, dtype=dtype)
        if value is not None: 
            if value is not klass.useDefault:
                value = numpy.asarray(value, klass.default.dtype)
            else: value = klass.default
            self.view(value.dtype)[:] = value

        #self.gldtype.configFrom(self)
        return self

    @classmethod
    def fromData(klass, data, dtype=None, copy=False):
        if isinstance(data, klass):
            dtype2 = data.dtype
            if dtype is None:
                self = (data if copy else data.copy())
            elif copy or dtype2 != dtype:
                self = data.astype(dtype)
            else: self = data

            #self.gldtype.configFrom(self)
            return self

        elif isinstance(data, ndarray):
            if dtype is None:
                intype = data.dtype
            else: intype = self.gldtype.dtypefmt(dtype)

            self = data.view(klass)
            if intype != data.dtype:
                self = self.astype(intype)
            elif copy: 
                self = self.copy()

            #self.gldtype.configFrom(self)
            return self

        else:
            return klass.fromDataRaw(data, dtype, copy)

    @classmethod
    def fromDataRaw(klass, data, dtype=None, copy=False):
        dtype, shape = klass.gldtype.lookupDTypeFrom(dtype, numpy.shape(data), True)
        self = ndarray.__new__(klass, shape, dtype=dtype)
        self[:] = data
        #self.gldtype.configFrom(self)
        return self



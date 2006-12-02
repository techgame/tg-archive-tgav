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
        'int8': gl.GL_BYTE,
        'uint16': gl.GL_UNSIGNED_SHORT,
        'int16': gl.GL_SHORT,
        'uint32': gl.GL_UNSIGNED_INT,
        'int32': gl.GL_INT,
        'float32': gl.GL_FLOAT,
        'float64': gl.GL_DOUBLE,
        }
    gltypeidUnmap = dict((v,k) for k,v in gltypeidMap.iteritems())

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

    def addFormats(self, dtypesList, entrySizes=(), default=NotImplemented):
        dtypeMap = self.dtypeMap
        for edtype in dtypesList:
            edtype = self.dtypefmt(edtype)

            if edtype.fields:
                dtypeMap[edtype.name, edtype.shape] = edtype
                print edtype

            elif edtype.shape or not entrySizes:
                dtypeMap[edtype.base.name, edtype.shape] = edtype
                print edtype

            else:
                for esize in entrySizes:
                    dt = numpy.dtype((edtype, esize))
                    dtypeMap[dt.base.name, dt.shape] = dt

        if default is not NotImplemented:
            self.defaultFormat = self.dtypefmt(default)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def configFrom(self, array, parent=None):
        if parent is not None:
            array.gltypeid = parent.gltypeid 
        else: array.gltypeid = self.gltypeidFor(array)

    def gltypeidFor(self, array):
        return self.gltypeidMap[array.dtype.base.name]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dtypeFrom = staticmethod(numpy.dtype)

    @classmethod
    def dtypefmt(klass, dtypefmt):
        if isinstance(dtypefmt, basestring) and (':' in dtypefmt):
            names, formats = zip(*[[i.strip() for i in p.split(':', 1)] for p in dtypefmt.split(',')])
            dtypefmt = dict(names=names, formats=formats)

        return klass.dtypeFrom(dtypefmt)

    def lookupDTypeFrom(self, dtype, shape, completeShape=False):
        print 'lookupDTypeFrom dtype: %r shape:%r complete: %r' % (dtype, shape, completeShape)
        if dtype is None:
            dtype = self.defaultFormat
        else:
            dtype = self.dtypefmt(dtype)
            key = (dtype.base.name, shape[-1:])

        if isinstance(shape, (int, long, float)):
            shape = (shape,)
        elif shape is None: shape = ()

        if completeShape:
            key = (dtype.base.name, shape[-1:])
            shape = shape[:-1]
        else:
            key = (dtype.base.name, dtype.shape)

        print '   ... key:', key
        dtype = self.dtypeMap[key]
        return dtype, shape


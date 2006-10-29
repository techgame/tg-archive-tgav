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

from ctypes import pythonapi

import numpy

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

bufferUsageMap = {
    gl.GL_STREAM_DRAW: gl.GL_STREAM_DRAW,
    'streamDraw': gl.GL_STREAM_DRAW,
    gl.GL_STREAM_READ: gl.GL_STREAM_READ,
    'streamRead': gl.GL_STREAM_READ,
    gl.GL_STREAM_COPY: gl.GL_STREAM_COPY,
    'streamCopy': gl.GL_STREAM_COPY,

    None: gl.GL_STATIC_DRAW,
    gl.GL_STATIC_DRAW: gl.GL_STATIC_DRAW,
    'staticDraw': gl.GL_STATIC_DRAW,
    gl.GL_STATIC_READ: gl.GL_STATIC_READ,
    'staticRead': gl.GL_STATIC_READ,
    gl.GL_STATIC_COPY: gl.GL_STATIC_COPY,
    'staticCopy': gl.GL_STATIC_COPY,

    gl.GL_DYNAMIC_DRAW: gl.GL_DYNAMIC_DRAW,
    'dynamicDraw': gl.GL_DYNAMIC_DRAW,
    gl.GL_DYNAMIC_READ: gl.GL_DYNAMIC_READ,
    'dynamicRead': gl.GL_DYNAMIC_READ,
    gl.GL_DYNAMIC_COPY: gl.GL_DYNAMIC_COPY,
    'dynamicCopy': gl.GL_DYNAMIC_COPY,
    }

accessMap = {
    gl.GL_READ_ONLY: gl.GL_READ_ONLY,
    'r': gl.GL_READ_ONLY,
    'read': gl.GL_READ_ONLY,

    gl.GL_WRITE_ONLY: gl.GL_WRITE_ONLY,
    'w': gl.GL_WRITE_ONLY,
    'write': gl.GL_WRITE_ONLY,

    None: gl.GL_READ_WRITE,
    gl.GL_READ_WRITE: gl.GL_READ_WRITE,
    'rw': gl.GL_READ_WRITE,
    'readWrite': gl.GL_READ_WRITE,
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BufferBase(object):
    _as_parameter_ = None # GLenum returned from glGenBuffers
    target = None
    size = 0
    dtype = numpy.ubyte
    accessByName = accessMap

    def __init__(self, **kw):
        self.set(kw)

    def create(self, usage=None):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")

        self.setUsage(usage)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    usageByName = bufferUsageMap
    _usage = usageByName[None]
    def getUsage(self):
        return self._usage
    def setUsage(self, usage):
        self._usage = self.usageByName[usage or self.usage]

        self._genId()
        self.bind()

    def release(self):
        self.unbind()
        self._delId()

    glGenBuffers = staticmethod(gl.glGenBuffers)
    def _genId(self):
        if self._as_parameter_ is None:
            p = gl.GLenum(0)
            self.glGenBuffers(1, byref(p))
            self._as_parameter_ = p
    glDeleteBuffers = staticmethod(gl.glDeleteBuffers)
    def _delId(self):
        p = self._as_parameter_
        if p is not None:
            self.glDeleteBuffers(1, byref(p))
            self._as_parameter_ = None

    glBindBuffer = staticmethod(gl.glBindBuffer)
    def bind(self):
        self.glBindBuffer(self.target, self)
    select = bind
    def unbind(self):
        self.glBindBuffer(self.target, 0)
    deselect = unbind

    glBufferData = staticmethod(gl.glBufferData)
    def sendData(self, data, usage=None):
        self.glBufferData(self.target, len(data), data, usage or self.usage)
        self.size = len(data)

    glBufferSubData = staticmethod(gl.glBufferData)
    def sendDataAt(self, data, offset):
        self.glBufferSubData(self.target, offset, len(data), data)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _bufferFromMemory = staticmethod(pythonapi.PyBuffer_FromMemory)
    _bufferFromReadWriteMemory = staticmethod(pythonapi.PyBuffer_FromReadWriteMemory)

    glMapBuffer = staticmethod(gl.glMapBuffer)
    _mapBuffer = None
    def map(self, access, dtype=None):
        access = self.accessByName[access]
        result = self._mapBuffer
        if result is None:
            ptr = self.glMapBuffer(self.target, access)

            if access == gl.GL_READ_ONLY:
                buf = self._bufferFromMemory(ptr, self.size)
            else:
                buf = self._bufferFromReadWriteMemory(ptr, self.size)

            result = numpy.frombuffer(buf, dtype or self.dtype)

            self._mapBuffer = result
            self._map_count = 1
            self._map_access = access

        else:
            if self._map_access not in (access, gl.GL_READ_WRITE,):
                raise Exception("Multiple MapBuffer access mismatch.  Origional: %s Request: %s" % (self._map_access, access))

            self._map_count += 1

            if dtype is None:
                return result
            else:
                return result.view(dtype)

    glUnmapBuffer = staticmethod(gl.glUnmapBuffer)
    def unmap(self):
        self._map_count -= 1
        if self._map_count <= 0:
            self.glUnmapBuffer(self.target)
            self._map_count = 0

    def isGLBuffer(self):
        return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBuffer(BufferBase):
    target = gl.GL_ARRAY_BUFFER

    def isGLArrayBuffer(self):
        return True

class ElementArrayBuffer(BufferBase):
    target = gl.GL_ELEMENT_ARRAY_BUFFER

    def isGLElementBuffer(self):
        return True


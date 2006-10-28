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
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BufferBase(object):
    _as_parameter_ = None # GLenum returned from glGenBuffers
    target = None

    size = 0
    dtype = numpy.ubyte

    usage = None
    usageByName = {
        gl.GL_STREAM_DRAW: gl.GL_STREAM_DRAW,
        'streamDraw': gl.GL_STREAM_DRAW,
        gl.GL_STREAM_READ: gl.GL_STREAM_READ,
        'streamRead': gl.GL_STREAM_READ,
        gl.GL_STREAM_COPY: gl.GL_STREAM_COPY,
        'streamCopy': gl.GL_STREAM_COPY,

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

    accessByName = {
        gl.GL_READ_ONLY: gl.GL_READ_ONLY,
        'r': gl.GL_READ_ONLY,
        'read': gl.GL_READ_ONLY,

        gl.GL_WRITE_ONLY: gl.GL_WRITE_ONLY,
        'w': gl.GL_WRITE_ONLY,
        'write': gl.GL_WRITE_ONLY,

        gl.GL_READ_WRITE: gl.GL_READ_WRITE,
        'rw': gl.GL_READ_WRITE,
        'readWrite': gl.GL_READ_WRITE,
        }

    def create(self, usage=None):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")

        self.usage = self.usageByName[usage or self.usage]

        self._genId()
        self.bind()

    def release(self):
        self.unbind()
        self._delId()

    def _genId(self):
        if self._as_parameter_ is None:
            p = gl.GLenum(0)
            gl.glGenBuffers(1, byref(p))
            self._as_parameter_ = p
    def _delId(self):
        p = self._as_parameter_
        if p is not None:
            gl.glDeleteBuffers(1, byref(p))
            self._as_parameter_ = None

    glBindBuffer = staticmethod(gl.glBindBuffer)
    def bind(self):
        self.glBindBuffer(self.target, self)
    def unbind(self):
        self.glBindBuffer(self.target, 0)

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
    def map(self, access):
        access = self.accessByName[access]
        result = self._mapBuffer
        if result is None:
            ptr = self.glMapBuffer(self.target, access)

            if access == gl.GL_READ_ONLY:
                buf = self._bufferFromMemory(ptr, self.size)
            else:
                buf = self._bufferFromReadWriteMemory(ptr, self.size)

            result = numpy.frombuffer(buf, self.dtype)

            self._mapBuffer = result
            self._map_count = 1
            self._map_access = access

        elif self._map_access is not access:
            raise Exception("Multiple MapBuffer access mismatch.  Origional: %s Request: %s" % (self._map_access, access))

        else:
            self._map_count += 1

        return result

    glUnmapBuffer = staticmethod(gl.glUnmapBuffer)
    def unmap(self):
        self._map_count -= 1
        if self._map_count <= 0:
            self.glUnmapBuffer(self.target)
            self._map_count = 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBuffer(BufferBase):
    target = gl.GL_ARRAY_BUFFER

class ElementArrayBuffer(BufferBase):
    target = gl.GL_ELEMENT_ARRAY_BUFFER


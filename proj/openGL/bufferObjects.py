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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BufferBase(object):
    _as_parameter_ = None # GLenum returned from glGenBuffers
    target = None
    usage = None
    usageByName = {
        'streamDraw': gl.GL_STREAM_DRAW,
        'streamRead': gl.GL_STREAM_READ,
        'streamCopy': gl.GL_STREAM_COPY,

        'staticDraw': gl.GL_STATIC_DRAW,
        'staticRead': gl.GL_STATIC_READ,
        'staticCopy': gl.GL_STATIC_COPY,

        'dynamicDraw': gl.GL_DYNAMIC_DRAW,
        'dynamicRead': gl.GL_DYNAMIC_READ,
        'dynamicCopy': gl.GL_DYNAMIC_COPY,
        }

    def create(self, usage=None):
        if not self._as_parameter_ is None:
            raise Exception("Create has already been called for this instance")
        usage = usage or self.usage
        usage = self.usageByName.get(usage, usage)
        self.usage = usage

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayBuffer(BufferBase):
    target = gl.GL_ARRAY_BUFFER

class ElementArrayBuffer(BufferBase):
    target = gl.GL_ELEMENT_ARRAY_BUFFER


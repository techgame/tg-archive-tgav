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

from functools import partial
from ..raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayViewFactory(object):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ArrayViewBase(object):
    drawMode = gl.GL_POINTS
    immediateFnMap = dict()

    def select(self, vboOffset=None):
        self.enable()
        self.bind(vboOffset)

    def enable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def glArrayPointer(self, count, dataFormat, stride, ptr):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def bind(self, vboOffset=None):
        elemSize = self.dtype[0].shape[-1]
        if vboOffset is None:
            self.glArrayPointer(elemSize, self.dataFormat, self.strides[-1], self.ctypes)
        else:
            self.glArrayPointer(elemSize, self.dataFormat, self.strides[-1], vboOffset)

    def deselect(self):
        self.unbind()
        self.disable()

    def disable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def unbind(self):
        pass

    glDrawArrays = staticmethod(gl.glDrawArrays)
    def draw(self, drawMode=None, vboOffset=None):
        self.select(vboOffset)
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)
    def drawRaw(self, drawMode=None):
        self.glDrawArrays(drawMode or self.drawMode, 0, self.size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def glImmediateFor(klass, glarray):
        gltypeid = glarray.gltypeid
        dim = glarray.shape[-1]
        return klass.immediateFnMap[gltypeid, dim]

    def glImmediate(self):
        raise NotImplementedError('_glImmediate_ should be repopulated from the dataformat and shape of the array')
    def _configImmediateFn(self):
        key = (self.dataFormat)#, self.dtype[0].shape[-1])
        glImmediateFn = self.immediateFnMap[key]
        self._glImmediate_ = partial(glImmediateFn, self.ctypes.data_as(glImmediateFn.api.argtypes[-1]))
    def _glImmediate_(self, ptr):
        raise NotImplementedError('_glImmediate_ should be repopulated from the dataformat and shape of the array')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexArrayView(ArrayViewBase):
    glArrayType = gl.GL_VERTEX_ARRAY
    glArrayPointer = staticmethod(gl.glVertexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class TexureCoordArrayView(ArrayViewBase):
    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexureCoordArrayView(ArrayViewBase):
    glClientActiveTexture = staticmethod(gl.glClientActiveTexture)
    texUnit = gl.GL_TEXTURE0

    def bind(self):
        self.glClientActiveTexture(self.texUnit)
        if vboOffset is None:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, self.ctypes)
        else:
            self.glArrayPointer(self.shape[-1], self.dataFormat, 0, vboOffset)
    def unbind(self):
        self.glClientActiveTexture(self.texUnit)
        self.glArrayPointer(3, self.dataFormat, 0, None)
    
    def glImmediate(self):
        self._glImmediate_(self.texUnit, self.ctypes)


class NormalArrayView(ArrayViewBase):
    glArrayType = gl.GL_NORMAL_ARRAY
    glArrayPointer = staticmethod(gl.glNormalPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class ColorArrayView(ArrayViewBase):
    glArrayType = gl.GL_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class SecondaryColorArrayView(ArrayViewBase):
    glArrayType = gl.GL_SECONDARY_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glSecondaryColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorIndexArrayView(ArrayViewBase):
    glArrayType = gl.GL_INDEX_ARRAY
    glArrayPointer = staticmethod(gl.glIndexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class FogCoordArrayView(ArrayViewBase):
    glArrayType = gl.GL_FOG_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glFogCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class EdgeFlagArrayView(ArrayViewBase):
    glArrayType = gl.GL_EDGE_FLAG_ARRAY
    glArrayPointer = staticmethod(gl.glEdgeFlagPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


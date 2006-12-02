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
    immediateFnMap = dict([])

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
    immediateFnMap = dict([
        ((gl.GL_SHORT, 2), gl.glVertex2sv),
        ((gl.GL_SHORT, 3), gl.glVertex3sv),
        ((gl.GL_SHORT, 4), gl.glVertex4sv),

        ((gl.GL_INT, 2), gl.glVertex2iv),
        ((gl.GL_INT, 3), gl.glVertex3iv),
        ((gl.GL_INT, 4), gl.glVertex4iv),

        ((gl.GL_FLOAT, 2), gl.glVertex2fv),
        ((gl.GL_FLOAT, 3), gl.glVertex3fv),
        ((gl.GL_FLOAT, 4), gl.glVertex4fv),

        ((gl.GL_DOUBLE, 2), gl.glVertex2dv),
        ((gl.GL_DOUBLE, 3), gl.glVertex3dv),
        ((gl.GL_DOUBLE, 4), gl.glVertex4dv),
        ])

    glArrayType = gl.GL_VERTEX_ARRAY
    glArrayPointer = staticmethod(gl.glVertexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class TexureCoordArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_SHORT, 1), gl.glTexCoord1sv),
        ((gl.GL_SHORT, 2), gl.glTexCoord2sv),
        ((gl.GL_SHORT, 3), gl.glTexCoord3sv),
        ((gl.GL_SHORT, 4), gl.glTexCoord4sv),

        ((gl.GL_INT, 1), gl.glTexCoord1iv),
        ((gl.GL_INT, 2), gl.glTexCoord2iv),
        ((gl.GL_INT, 3), gl.glTexCoord3iv),
        ((gl.GL_INT, 4), gl.glTexCoord4iv),

        ((gl.GL_FLOAT, 1), gl.glTexCoord1fv),
        ((gl.GL_FLOAT, 2), gl.glTexCoord2fv),
        ((gl.GL_FLOAT, 3), gl.glTexCoord3fv),
        ((gl.GL_FLOAT, 4), gl.glTexCoord4fv),

        ((gl.GL_DOUBLE, 1), gl.glTexCoord1dv),
        ((gl.GL_DOUBLE, 2), gl.glTexCoord2dv),
        ((gl.GL_DOUBLE, 3), gl.glTexCoord3dv),
        ((gl.GL_DOUBLE, 4), gl.glTexCoord4dv),
        ])

    glArrayType = gl.GL_TEXTURE_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glTexCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class MultiTexureCoordArrayView(ArrayViewBase):
    glClientActiveTexture = staticmethod(gl.glClientActiveTexture)
    texUnit = gl.GL_TEXTURE0

    immediateFnMap = [
        ((gl.GL_SHORT, 1), gl.glMultiTexCoord1sv),
        ((gl.GL_SHORT, 2), gl.glMultiTexCoord2sv),
        ((gl.GL_SHORT, 3), gl.glMultiTexCoord3sv),
        ((gl.GL_SHORT, 4), gl.glMultiTexCoord4sv),

        ((gl.GL_INT, 1), gl.glMultiTexCoord1iv),
        ((gl.GL_INT, 2), gl.glMultiTexCoord2iv),
        ((gl.GL_INT, 3), gl.glMultiTexCoord3iv),
        ((gl.GL_INT, 4), gl.glMultiTexCoord4iv),

        ((gl.GL_FLOAT, 1), gl.glMultiTexCoord1fv),
        ((gl.GL_FLOAT, 2), gl.glMultiTexCoord2fv),
        ((gl.GL_FLOAT, 3), gl.glMultiTexCoord3fv),
        ((gl.GL_FLOAT, 4), gl.glMultiTexCoord4fv),

        ((gl.GL_DOUBLE, 1), gl.glMultiTexCoord1dv),
        ((gl.GL_DOUBLE, 2), gl.glMultiTexCoord2dv),
        ((gl.GL_DOUBLE, 3), gl.glMultiTexCoord3dv),
        ((gl.GL_DOUBLE, 4), gl.glMultiTexCoord4dv),
        ]

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
    immediateFnMap = dict([
        ((gl.GL_BYTE, 3), gl.glNormal3bv),
        ((gl.GL_SHORT, 3), gl.glNormal3sv),
        ((gl.GL_INT, 3), gl.glNormal3iv),
        ((gl.GL_FLOAT, 3), gl.glNormal3fv),
        ((gl.GL_DOUBLE, 3), gl.glNormal3dv),
        ])

    glArrayType = gl.GL_NORMAL_ARRAY
    glArrayPointer = staticmethod(gl.glNormalPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class ColorArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_UNSIGNED_BYTE, 3), gl.glColor3ubv),
        ((gl.GL_UNSIGNED_BYTE, 4), gl.glColor4ubv),

        ((gl.GL_UNSIGNED_SHORT, 3), gl.glColor3usv),
        ((gl.GL_UNSIGNED_SHORT, 4), gl.glColor4usv),

        ((gl.GL_UNSIGNED_INT, 3), gl.glColor3uiv),
        ((gl.GL_UNSIGNED_INT, 4), gl.glColor4uiv),

        ((gl.GL_BYTE, 3), gl.glColor3bv),
        ((gl.GL_BYTE, 4), gl.glColor4bv),

        ((gl.GL_SHORT, 3), gl.glColor3sv),
        ((gl.GL_SHORT, 4), gl.glColor4sv),

        ((gl.GL_INT, 3), gl.glColor3iv),
        ((gl.GL_INT, 4), gl.glColor4iv),

        ((gl.GL_FLOAT, 3), gl.glColor3fv),
        ((gl.GL_FLOAT, 4), gl.glColor4fv),

        ((gl.GL_DOUBLE, 3), gl.glColor3dv),
        ((gl.GL_DOUBLE, 4), gl.glColor4dv),
        ])

    glArrayType = gl.GL_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class SecondaryColorArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_UNSIGNED_BYTE, 3), gl.glSecondaryColor3ubv),
        ((gl.GL_UNSIGNED_SHORT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_UNSIGNED_INT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_BYTE, 3), gl.glSecondaryColor3usv),
        ((gl.GL_SHORT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_INT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_FLOAT, 3), gl.glSecondaryColor3usv),
        ((gl.GL_DOUBLE, 3), gl.glSecondaryColor3usv),
        ])

    glArrayType = gl.GL_SECONDARY_COLOR_ARRAY
    glArrayPointer = staticmethod(gl.glSecondaryColorPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))

class ColorIndexArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_UNSIGNED_BYTE, 1), gl.glIndexubv),
        ((gl.GL_SHORT, 1), gl.glIndexsv),
        ((gl.GL_INT, 1), gl.glIndexiv),
        ((gl.GL_FLOAT, 1), gl.glIndexfv),
        ((gl.GL_DOUBLE, 1), gl.glIndexdv),
        ])

    glArrayType = gl.GL_INDEX_ARRAY
    glArrayPointer = staticmethod(gl.glIndexPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class FogCoordArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_FLOAT, 1), gl.glFogCoordfv),
        ((gl.GL_DOUBLE, 1), gl.glFogCoorddv),
        ])

    glArrayType = gl.GL_FOG_COORD_ARRAY
    glArrayPointer = staticmethod(gl.glFogCoordPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


class EdgeFlagArrayView(ArrayViewBase):
    immediateFnMap = dict([
        ((gl.GL_UNSIGNED_BYTE, 1), gl.glEdgeFlagv),
        ])

    glArrayType = gl.GL_EDGE_FLAG_ARRAY
    glArrayPointer = staticmethod(gl.glEdgeFlagPointer)
    enable = staticmethod(partial(gl.glEnableClientState, glArrayType))
    disable = staticmethod(partial(gl.glDisableClientState, glArrayType))


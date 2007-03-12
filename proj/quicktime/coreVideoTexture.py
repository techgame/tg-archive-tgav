##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2007  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import ctypes, ctypes.util
from ctypes import c_void_p

import numpy

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libCoreVideoPath = ctypes.util.find_library("CoreVideo")
libCoreVideo = ctypes.cdll.LoadLibrary(libCoreVideoPath)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CVOpenGLTexture(object):
    size = None
    texCoords = None
    _as_parameter_ = 0
    target = None

    def __init__(self):
        self.texCoords = numpy.zeros((4,2), 'f')
        self._texCoordsAddresses = [tc.ctypes.data for tc in self.texCoords]
        assert len(self._texCoordsAddresses) == 4

        self._cvTextureRef = c_void_p(0)

    def updateCVTexture(self, cvTextureRef):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def update(self, force=False):
        if not force and not self.isNewImageAvailable():
            return False

        libCoreVideo.CVOpenGLTextureRelease(self._cvTextureRef)
        cvTextureRef = c_void_p(0)
        self._cvTextureRef = cvTextureRef

        self.updateCVTexture(cvTextureRef)

        self.target = libCoreVideo.CVOpenGLTextureGetTarget(cvTextureRef)
        self._as_parameter_ = libCoreVideo.CVOpenGLTextureGetName(cvTextureRef)

        # CVOpenGLTextureGetCleanTexCoords takes into account whether or not the texture is flipped
        libCoreVideo.CVOpenGLTextureGetCleanTexCoords(cvTextureRef, *self._texCoordsAddresses)
        #libCoreVideo.CVOpenGLTextureIsFlipped(cvTextureRef)

        self.size = abs(self.texCoords[2]-self.texCoords[0])
        return True

    def renderDirect(self):
        if not self._as_parameter_:
            return False

        texCoords = self.texCoords
        w, h = texCoords[2]-texCoords[0]
        gl.glPushMatrix()
        if h<0:
            gl.glTranslatef(0, -h, 0)
            gl.glScalef(1, -1, 1)

        self.select()
        gl.glBegin(gl.GL_QUADS)

        gl.glTexCoord2f(*texCoords[0])
        gl.glVertex2f(*texCoords[0])

        gl.glTexCoord2f(*texCoords[1])
        gl.glVertex2f(*texCoords[1])

        gl.glTexCoord2f(*texCoords[2])
        gl.glVertex2f(*texCoords[2])

        gl.glTexCoord2f(*texCoords[3])
        gl.glVertex2f(*texCoords[3])

        gl.glEnd()

        gl.glPopMatrix()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def bind(self):
        gl.glBindTexture(self.target, self)
    def unbind(self):
        gl.glBindTexture(self.target, 0)

    def enable(self):
        gl.glEnable(self.target)
    def disable(self):
        gl.glDisable(self.target)

    def select(self, unit=None):
        if unit is not None:
            gl.glActiveTexture(unit)
        self.bind()
        self.enable()
        return self

    def deselect(self):
        self.disable()
        self.unbind()
        return self


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

from TG.observing import ObservableObject

from TG.openGL.raw import gl
from TG.openGL.raw.gl import *
from TG.openGL.data import Vector, Color

from TG.openGL.data.image import ImageTexture2d

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageObject(ObservableObject):
    ImageTexture = ImageTexture2d

    roundValues = True

    def __init__(self, image=None, format=True, **kwattr):
        self.imageTex = self.ImageTexture()
        if kwattr: 
            self.set(kwattr)
        if image is not None:
            self.load(image, format)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    color = Color.property('#f')
    pos = Vector.property([0,0,0])
    size = Vector.property([0,0,0])
    align = Vector.property([.5, .5, 0])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self, **kwattr):
        if kwattr: 
            self.set(kwattr)

        imageTex = self.imageTex
        texCoords = imageTex.texCoordsForImage()
        geom = imageTex.verticesForImage()

        self.size = imageTex.imageSize
        off = self.pos - (self.align*self.size)
        if self.roundValues:
            off = off.round()
        geom += off.round()

        self.texCoords = texCoords
        self.glTexCoordv = texCoords.glinfo.glImmediateFor(texCoords)
        texCoordsArrPtr = texCoords.glinfo.glArrayPointer
        self.glTexCoordsArrPtr = partial(texCoordsArrPtr, 
                texCoords.shape[-1],
                texCoords.glTypeId,
                texCoords.strides[-1]*texCoords.shape[-1],
                texCoords.ctypes.data_as(texCoordsArrPtr.api.argtypes[-1]))
        self.glEnableTexCoordArray = partial(gl.glEnableClientState, texCoords.glinfo.glKindId)

        self.geom = geom
        geomArrPtr = geom.glinfo.glArrayPointer
        self.glVertexv = geom.glinfo.glImmediateFor(geom)
        self.glGeomArrPtr = partial(geomArrPtr, 
                geom.shape[-1],
                geom.glTypeId,
                geom.strides[-1]*geom.shape[-1],
                geom.ctypes.data_as(geomArrPtr.api.argtypes[-1]))
        self.glEnableGeomArray = partial(gl.glEnableClientState, geom.glinfo.glKindId)

        self.glDrawArrays = partial(gl.glDrawArrays, gl.GL_QUADS, 0, 4)
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load(self, image=None, format=True, **kwattr):
        self.imageTex.load(image, format)
        if kwattr: 
            self.set(kwattr)
        self.update()
    def loadImage(self, image=None, format=True, **kwattr):
        self.imageTex.loadImage(image, format)
        if kwattr: 
            self.set(kwattr)
        self.update()
    def loadFilename(self, image=None, format=True, **kwattr):
        self.imageTex.loadFilename(image, format)
        if kwattr: 
            self.set(kwattr)
        self.update()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def render(self):
        gl.glColor4f(*self.color.flat)

        self.imageTex.select()

        if 0:
            self.glEnableTexCoordArray()
            self.glTexCoordsArrPtr()

            self.glEnableGeomArray()
            self.glGeomArrPtr()

            self.glDrawArrays()
        else:
            gl.glBegin(gl.GL_QUADS)
            for t, v in zip(self.texCoords, self.geom):
                self.glTexCoordv(t.ctypes.data_as(self.glTexCoordv.api.argtypes[-1]))
                self.glVertexv(v.ctypes.data_as(self.glVertexv.api.argtypes[-1]))
            gl.glEnd()


#!/usr/bin/env python
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

from TG.openGL.texture import Texture

from TG.openGL.raw import gl, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureBase(Texture):
    target = None
    format = gl.GL_ALPHA

    texParams = Texture.texParams.copy()
    texParams.update(
            wrap=gl.GL_CLAMP,
            genMipmaps=True,
            magFilter=gl.GL_LINEAR,
            minFilter=gl.GL_LINEAR_MIPMAP_LINEAR,
            )
    dataFormat = gl.GL_ALPHA
    dataType = gl.GL_UNSIGNED_BYTE

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def createMosaic(self, mosaicSize):
        self.select()
        size = self.validSizeForTarget(mosaicSize)
        data = self.blankImage2d(size=size, format=self.dataFormat, dataType=self.dataType)
        data.newPixelStore(alignment=1, rowLength=0)
        self._mosaicData = data
        return data

    def renderGlyph(self, glyph, pos, size):
        texData = self._mosaicData

        glyph.render()

        texData.texCData(glyph.bitmap.buffer, dict(rowLength=glyph.bitmap.pitch))
        texData.setSubImageOn(self, pos=pos, size=size)
        return self.texCoordsFrom(pos, size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def texCoordsFrom(self, pos, size, totalSize):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureRect(FontTextureBase):
    target = glext.GL_TEXTURE_RECTANGLE_ARB

    def texCoordsFrom(self, (x0, y0), (w, h)):
        x1 = x0 + w; y1 = y0 + h
        return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]

FontTexture = FontTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTexture2D(FontTextureBase):
    target = gl.GL_TEXTURE_2D

    def texCoordsFrom(self, (x0, y0), (w, h)):
        (tw, th) = self.size

        x1 = (x0 + w)/tw; x0 /= tw
        y1 = (y0 + h)/th; y0 /= th
        return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


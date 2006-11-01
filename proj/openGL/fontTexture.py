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

import string
import numpy

from TG.openGL.texture import Texture
from TG.openGL.blockMosaic import BlockMosaicAlg

from TG.openGL.raw import gl, glu, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureBase(Texture):
    LayoutAlgorithm = BlockMosaicAlg

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

    def renderFace(self, ftFace, iterableGlyphIndexes):
        size, layout = self._layoutMosaic(iterableGlyphIndexes, ftFace)
        size = self.validSizeForTarget(size)

        data = self._blankTexture(size)
        texCoordMapping = self._renderMosaic(data, layout, ftFace)
        return texCoordMapping

    def _blankTexture(self, size):
        data = self.data2d(size=size, format=self.dataFormat, dataType=self.dataType)
        data.texBlank()
        data.setImageOn(self)
        return data

    def _layoutMosaic(self, iterableGlyphIndexes, ftFace):
        maxSize = self.getMaxTextureSize()
        alg = self.LayoutAlgorithm((maxSize, maxSize))

        # make sure we have the "unknown" glyph
        glyphIndex = 0
        size = ftFace.loadGlyph(glyphIndex).bitmapSize
        alg.addBlock(size, key=glyphIndex)

        # add all the other glyph indexes
        for glyphIndex in iterableGlyphIndexes:
            if glyphIndex:
                size = ftFace.loadGlyph(glyphIndex).bitmapSize
                alg.addBlock(size, key=glyphIndex)

        usedSize, glyphLayout, unplacedGlyphs = alg.layout()

        if unplacedGlyphs:
            raise Exception("Not all characters could be placed in mosaic")

        return usedSize, glyphLayout

    def _renderMosaic(self, data, glyphLayout, ftFace):
        mapping = []
        data.newPixelStore(alignment=1, rowLength=0)

        loadGlyph = ftFace.loadGlyph
        texCoordsFromPosSize = self.texCoordsFromPosSize
        totalSize = float(self.size[0]), float(self.size[1])
        for block in glyphLayout:
            glyphIndex = block.key; pos = block.pos; size = block.size;

            glyph = loadGlyph(glyphIndex)
            glyph.render()

            data.texCData(glyph.bitmap.buffer, dict(rowLength=glyph.bitmap.pitch))
            data.setSubImageOn(self, pos=pos, size=size)

            mapping.append((glyphIndex, texCoordsFromPosSize(pos, size, totalSize)))

        data.texClear()
        return mapping

    def texCoordsFromPosSize(self, pos, size, totalSize):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureRect(FontTextureBase):
    target = glext.GL_TEXTURE_RECTANGLE_ARB

    def texCoordsFromPosSize(self, (x0, y0), (w, h), (tw, th)):
        x1 = x0 + w; y1 = y0 + h
        return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]

FontTexture = FontTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTexture2D(FontTextureBase):
    target = gl.GL_TEXTURE_2D

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def texCoordsFromPosSize(self, (x0, y0), (w, h), (tw, th)):
        x1 = (x0 + w)/tw; x0 /= tw
        y1 = (y0 + h)/th; y0 /= th
        return [(x0, y1), (x1, y1), (x1, y0), (x0, y0)]


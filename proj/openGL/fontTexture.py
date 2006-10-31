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

from TG.freetype2.face import FreetypeFace

from TG.openGL.texture import Texture
from TG.openGL.blockMosaic import BlockMosaicAlg
from TG.openGL.vertexArray import TexureCoordArray

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
    pointSize = (1./64., 1./64.)

    vertexDataFormat = 'f'
    texCoordDataFormat = 'f'

    @classmethod
    def fromFace(klass, ftFace):
        self = klass()
        self.renderFace(ftFace)
        return self

    @classmethod
    def fromFilename(klass, filename, size, dpi=None):
        ftFace = FreetypeFace(filename)
        ftFace.setSize(size, dpi)
        return klass.fromFace(ftFace)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderFace(self, ftFace, charset=string.printable):
        size, layout = self._layoutMosaic(charset, ftFace)
        size = self.validSizeForTarget(size)

        data = self._blankTexture(size)
        self._renderMosaic(data, layout, ftFace)

    def _blankTexture(self, size):
        data = self.data2d(size=size, format=self.dataFormat, dataType=self.dataType)
        data.texBlank()
        data.setImageOn(self)
        return data

    def _layoutMosaic(self, charset, ftFace):
        maxSize = self.getMaxTextureSize()
        alg = self.LayoutAlgorithm((maxSize, maxSize))

        for glyphIndex, glyph in ftFace.iterUniqueGlyphs(charset):
            alg.addBlock(glyph.bitmapSize, key=glyphIndex)

        usedSize, glyphLayout, unplacedGlyphs = alg.layout()

        if unplacedGlyphs:
            raise Exception("Not all characters could be placed in mosaic")

        return usedSize, glyphLayout

    def _renderMosaic(self, data, glyphLayout, ftFace):
        data.newPixelStore(alignment=1, rowLength=0)

        for block in glyphLayout:
            glyph = ftFace.loadGlyph(block.key)
            glyph.render()
            data.texCData(glyph.bitmap.buffer, dict(rowLength=glyph.bitmap.pitch))
            data.setSubImageOn(self, pos=block.pos, size=block.size)

        data.texClear()

    def _genTexCoords(self, glyphLayout, ftFace):
        mapping = {}
        arraySize = (len(glyphLayout), 4, 2)
        vertexEntrySize = arraySize[1] * arraySize[2]

        verticies = numpy.array(arraySize, self.vertexDataFormat)
        advance = numpy.array((len(glyphLayout), 2), self.vertexDataFormat)
        texCoords = numpy.array(arraySize, self.texCoordDataFormat)

        getTexCoordsFor = self._getTexCoordsFor
        getVerticisFor = self._getVerticiesFor

        loadGlyph = ftFace.loadGlyph

        tw, th = self.size
        ptw, pth = self.pointSize
        for i, (glyphIndex, block) in enumerate(glyphLayout.iteritems()):
            mapping[glyphIndex] = i * vertexEntrySize
            texCoords[i] = getTexCoordsFor(block, tw, th)

            glyph = loadGlyph(glyphIndex)
            verticies[i] = getVerticisFor(glyph, ptw, pth)
            advance[i] = getAdvanceFor(glyph, ptw, pth)

    def _getTexCoordsFor(self, block, tw, th):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def _getVerticiesFor(self, glyph, ptw, pth):
        m = glyph.metrics
        x0 = (m.horiBearingX) / ptw
        y0 = (m.horiBearingY - m.height) / pth

        x1 = (m.horiBearingX + m.width) / ptw
        y1 = (m.horiBearingY) / pth

        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _getAdvanceFor(self, glyph, ptw, pth):
        ax, ay = glyph.advance
        return (ax/ptw, ay/pth)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureRect(FontTextureBase):
    target = glext.GL_TEXTURE_RECTANGLE_ARB

    def _getTexCoordsFor(self, block, tw, th):
        x0, y0 = block.pos
        x1 = x0 + block.size[0]
        y1 = y0 + block.size[1]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

FontTexture = FontTextureRect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTexture2D(FontTextureBase):
    target = gl.GL_TEXTURE_2D

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _getTexCoordsFor(self, block, tw, th):
        x0, y0 = block.pos
        x1 = x0 + block.size[0]
        y1 = y0 + block.size[1]
        x0 /= tw; y0 /= th; x1 /= tw; y1 /= th;
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


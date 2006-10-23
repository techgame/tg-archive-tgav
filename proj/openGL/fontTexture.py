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
from TG.openGL.raw import gl, glu, glext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextureBase(Texture):
    LayoutAlgorithm = BlockMosaicAlg
    texParams = dict(
            #target=0,
            format=gl.GL_ALPHA,
            wrap=gl.GL_CLAMP,

            genMipmaps=True,
            filter=(gl.GL_LINEAR, gl.GL_LINEAR_MIPMAP_LINEAR),
            )
    dataFormat = gl.GL_ALPHA
    dataType = gl.GL_UNSIGNED_BYTE
    pointSize = (1./64., 1./64.)

    def __init__(self, ftFace):
        Texture.__init__(self, **self.texParams)
        self._setupMipmaps()
        self.ftFace = ftFace

    @classmethod
    def fromFont(klass, ftFont):
        return klass(ftFont.face)

    @classmethod
    def fromFilenameAndSize(klass, filename, size, dpi=None):
        ftFace = FreetypeFace(filename)
        ftFace.setSize(size, dpi)
        return klass(ftFace)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _setupMipmaps(self):
        self.set(genMipmaps=True, filter=self.texFilter)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _size = None
    def getSize(self):
        return self._size
    def setSize(self, size):
        if size == self._size:
            return
        self._size = size
        self._invalidate()

    def _setLayoutSize(self, size):
        self._size = size
    size = property(getSize, setSize)

    _chars = string.printable
    def getChars(self):
        return self._chars
    def setChars(self, chars):
        if chars == self._chars:
            return
        self._chars = chars
        self._invalidate()
    chars = property(getChars, setChars)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _invalidate(self):
        self._clear()
        size, layout = self._layoutMosaic(self.getChars(), self.ftFace)
        self._setLayoutSize(size)
        self._renderMosaic(layout, ftFace)

    def _clear(self):
        self.data = texture.data2d(size=self.size, format=self.dataFormat, dataType=self.dataType)
        self.data.texBlank()
        self.data.setImageOn(self)
        self.data.texClear()
        if gl.glGetError(): 
            raise RuntimeError("GL: %s" % (gl.glGetError(),))
        return self.data

    def _layoutMosaic(self, chars, ftFace):
        alg = self.LayoutAlgorithm(self.getMaxTextureSize())

        chars = u'\x00 \t\n\r' + chars
        self._charMap = ftFace.charIndexMap(chars)

        for glyphIndex, glyph in ftFace.iterUniqueGlyphs(chars):
            alg.addBlock(glyph.bitmapSize, key=glyphIndex)

        usedSize, glyphLayout, unplacedGlyphs = alg.layout()

        if unplacedGlyphs:
            raise Exception("Not all characters could be placed in mosaic")

        return usedSize, glyphLayout

    def _renderMosaic(self, glyphLayout, ftFace):
        data = self.data
        data.newPixelStore(alignment=1, rowLength=0)

        for block in glyphLayout:
            glyph = ftFace.loadGlyph(block.key)
            glyph.render()
            data.texCData(glyph.bitmap.buffer, dict(rowLength=glyph.bitmap.pitch))
            data.setSubImageOn(texture, pos=block.pos, size=block.size)

        data.texClear()

    vertexDataFormat = 'f'
    texCoordDataFormat = 'f'
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
            verticies[i] = getVerticisFor(glyph)
            advance[i] = glyph.advance

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
    texParams = FontTextureBase.copy()
    texParams.update(target=gl.GL_TEXTURE_RECTANGLE_ARB)

    def _getTexCoordsFor(self, block, tw, th):
        x0, y0 = block.pos
        x1 = x0 + block.size[0]
        y1 = y0 + block.size[1]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTexture2D(FontTextureBase):
    texParams = FontTextureBase.copy()
    texParams.update(target=gl.GL_TEXTURE_2D)

    def setSize(self, size):
        size = tuple(map(Texture.nextPowerOf2, size))
        return FontTextureBase.setSize(self, size)
    size = property(FontTextureBase.getSize, setSize)

    def _setLayoutSize(self, size):
        size = tuple(map(Texture.nextPowerOf2, size))
        return FontTextureBase._setLayoutSize(self, size)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _getTexCoordsFor(self, block, tw, th):
        x0, y0 = block.pos
        x1 = x0 + block.size[0]
        y1 = y0 + block.size[1]
        x0 /= tw; y0 /= th; x1 /= tw; y1 /= th;
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


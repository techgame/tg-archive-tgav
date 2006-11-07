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

import itertools

from numpy import ndarray, float32

from TG.freetype2.face import FreetypeFace

from TG.openGL import blockMosaic
from TG.openGL import fontTexture
from TG.openGL import interleavedArrays

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontGeometryArray(interleavedArrays.InterleavedArrays):
    dataFormat = gl.GL_T2F_V3F
    @classmethod
    def fromCount(klass, count):
        return klass.fromFormat((count, 4), klass.dataFormat)

class FontAdvanceArray(ndarray):
    @classmethod
    def fromCount(klass, count, dtype=float32):
        return klass((count, 4, 3), dtype)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ITextLayout(object):
    def layout(self, text):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

class SimpleTextLayout(ITextLayout):
    def layout(self, text):
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextTranslator(object):
    font = None
    texture = None
    _fontCharMap = None
    _fontAdvance = None
    _fontGeometry = None

    def __init__(self, text):
        self.text = text
        idx = map(self._fontCharMap.get, '\0' + self.text)
        self.indexes = idx
        self.advance = self._fontAdvance[idx]
        self.geometry = self._fontGeometry[idx]

    @classmethod
    def subclassFromFont(klass, font):
        kvars = dict(font=font, texture=font.texture, _fontCharMap=font.charMap, _fontAdvance=font.advance, _fontGeometry=font.geometry)
        subklass = type(klass)(klass.__name__+'_T_', (klass,), kvars)
        return subklass

    def layout(self):
        adv = self.advance.cumsum(0)
        geo = self.geometry[1:]
        geo['v'] += adv[:-1]
        self.geo = geo

        return geo, adv[-1], self.render

    def render(self):
        self.texture.select()
        self.geo.draw(gl.GL_QUADS)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Font(object):
    LayoutAlgorithm = blockMosaic.BlockMosaicAlg
    FontTexture = fontTexture.FontTexture
    FontGeometryArray = FontGeometryArray
    FontAdvanceArray = FontAdvanceArray
    TextTranslatorSubclassFromFont = TextTranslator.subclassFromFont
    TextTranslator = TextTranslator

    texture = None

    charMap = None
    geometry = None
    advance = None

    pointSize = (1./64., 1./64.)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, face, charset=None):
        if face is not None:
            self.compile(face, charset)

    @classmethod
    def fromFilename(klass, filename, size, dpi=None, charset=None):
        face = FreetypeFace(filename)
        face.setSize(size, dpi)
        return klass(face, charset)

    def translate(self, text):
        return self.TextTranslator(text)
    
    def layout(self, text):
        xlate = self.translate(text)
        return xlate.layout()
    def render(self, text):
        return self.layout(text)[-1]()

    def compile(self, face, charset):
        self.face = face
        self.lineHeight = face.lineHeight/self.pointSize[1]

        charMap = {'\0': 0}
        self.charMap = charMap

        gidxMap = {}
        aidxCounter = itertools.count(1)
        for char, gidx in face.iterCharIndexes(charset, True):
            aidx = aidxCounter.next()
            charMap.setdefault(char, aidx)
            gidxMap.setdefault(gidx, aidx)
        count = aidxCounter.next()

        # create a texture for the font mosaic, and run the mosaic algorithm
        mosaic = self._compileTexture(face, gidxMap)
        self._compileData(face, count, gidxMap, mosaic)

        self.TextTranslator = self.TextTranslatorSubclassFromFont(self)

    def _compileTexture(self, face, gidxMap):
        texture = self.FontTexture()
        mosaic, mosaicSize = self._compileGlyphMosaic(face, gidxMap, texture.getMaxTextureSize())
        texture.createMosaic(mosaicSize)
        self.texture = texture
        return mosaic

    def _compileGlyphMosaic(self, face, gidxMap, maxSize):
        alg = self.LayoutAlgorithm((maxSize, maxSize))

        mosaic = {}
        for gidx in gidxMap.iterkeys():
            size = face.loadGlyph(gidx).bitmapSize
            mosaic[gidx] = alg.addBlock(size)

        mosaicSize, layout, unplaced = alg.layout()

        if unplaced:
            raise RuntimeError("Not all characters could be placed in mosaic")

        return mosaic, mosaicSize

    def _compileData(self, face, count, gidxMap, mosaic):
        # create the result arrays
        geometry = self.FontGeometryArray.fromCount(count)
        self.geometry = geometry
        advance = self.FontAdvanceArray.fromCount(count)
        self.advance = advance

        # cache some methods
        loadGlyph = face.loadGlyph
        verticesFrom = self._verticesFrom
        advanceFrom = self._advanceFrom
        renderGlyph = self.texture.renderGlyph

        # cache some variables
        pointSize = self.pointSize

        # entry 0 is zero widht and has null geometry
        geometry[0]['t'] = 0.
        geometry[0]['v'] = 0.
        advance[0] = 0.
        # record the geometry and advance for each glyph, and render to the mosaic
        for gidx, aidx in gidxMap.iteritems():
            glyph = loadGlyph(gidx)

            geoEntry = geometry[aidx]
            geoEntry['v'] = verticesFrom(glyph.metrics, pointSize)
            advance[aidx,:] = advanceFrom(glyph.advance, pointSize)

            block = mosaic.get(gidx)
            if block is None: 
                geoEntry['t'] = 0.0
            else: 
                geoEntry['t'] = renderGlyph(glyph, block.pos, block.size)

    def _verticesFrom(self, metrics, (ptw, pth)):
        x0 = (metrics.horiBearingX) * ptw
        y0 = (metrics.horiBearingY - metrics.height) * pth

        x1 = (metrics.horiBearingX + metrics.width) * ptw
        y1 = (metrics.horiBearingY) * pth

        return [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)]

    def _advanceFrom(self, advance, (ptw, pth)):
        return [(advance[0]*ptw, advance[1]*pth, 0.)]*4


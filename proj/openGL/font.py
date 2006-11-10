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

from numpy import ndarray, float32, asarray

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
        return klass((count, 1, 3), dtype)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Font(object):
    LayoutAlgorithm = blockMosaic.BlockMosaicAlg
    FontTexture = fontTexture.FontTexture
    FontGeometryArray = FontGeometryArray
    FontAdvanceArray = FontAdvanceArray

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
        return map(self.charMap.get, text)

    def kernIndexes(self, idx, default=[0., 0., 0.]):
        km = self.kerningMap
        if not km or len(idx) < 2:
            return None
        
        r = asarray([km.get(e, default) for e in zip(idx, idx[1:])], float32)
        return r.reshape((-1, 1, 3))
    
    def compile(self, face, charset):
        self.face = face
        #face.printInfo()

        self.lineAdvance = self.FontAdvanceArray.fromCount(1)
        self.lineAdvance[0] = [0., -face.lineHeight*self.pointSize[1], 0.]

        charMap = {'\0': 0, '\n': 0, '\r': 0}
        self.charMap = charMap

        shane = {}
        gidxMap = {}
        aidxCounter = itertools.count(1)
        for char, gidx in face.iterCharIndexes(charset, True):
            aidx = aidxCounter.next()
            charMap.setdefault(char, aidx)
            gidxMap.setdefault(gidx, aidx)
            shane[gidx] = char
        count = aidxCounter.next()

        # create a texture for the font mosaic, and run the mosaic algorithm
        self._compileKerningMap(face, gidxMap, shane)
        mosaic = self._compileTexture(face, gidxMap)
        self._compileData(face, count, gidxMap, mosaic)

    def _compileKerningMap(self, face, gidxMap, shane):
        ptw, pth = self.pointSize
        kerningMap = {}
        gi = gidxMap.items()
        getKerningByIndex = face.getKerningByIndex
        for l, li in gi:
            for r, ri in gi:
                k = getKerningByIndex(l, r)
                if k[0] or k[1]:
                    k = [k[0]*ptw, k[1]*pth, 0.]
                    kerningMap[(li,ri)] = k
                    print (shane[l], shane[r]), '->', k
        self.kerningMap = kerningMap

    def _compileTexture(self, face, gidxMap):
        if self.FontTexture is None:
            return {}

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
        if self.texture is not None:
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
        return [(advance[0]*ptw, advance[1]*pth, 0.)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextData(object):
    font = None
    texture = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def factoryFor(klass, font):
        subklass = type(klass)(klass.__name__+'_T_', (klass,), {})
        subklass.setupClassFont(font)
        return subklass

    def setupFont(self, font):
        self.font = font
        self.texture = font.texture
        if not isinstance(self, type):
            self._recache()
    setupClassFont = classmethod(setupFont)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, text, font=None):
        if font is not None:
            self.setupFont(font)
        self.text = text

    _text = ""
    def getText(self):
        return self._text
    def setText(self, text):
        self._text = text
        self._recache()
    text = property(getText, setText)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _recache(self):
        if self.font is not None:
            self._xidx = self.font.translate(self.text)
        else:
            self._xidx = None
        self._advance = None
        self._lineAdvance = None
        self._geometry = None
        self._offset = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _lineAdvance = None
    def getLineAdvance(self):
        r = self._lineAdvance
        if r is None:
            r = self.font.lineAdvance
            self._lineAdvance = r
        return r
    lineAdvance = property(getLineAdvance)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _advance = None
    def getAdvance(self):
        r = self._advance
        if r is None:
            r = self.font.advance[[0]+self._xidx]
            k = self.font.kernIndexes(self._xidx)
            if k is not None:
                r[1:-1] += k
            self._advance = r
        return r

    def getPreAdvance(self):
        return self.getAdvance()[:-1]
    def getPostAdvance(self):
        return self.getAdvance()[1:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _offset = None
    def getOffset(self):
        r = self._offset
        if r is None:
            r = self.getAdvance().cumsum(0)
            self._offset = r

        return r

    def getOffsetAtStart(self):
        return self.getOffset()[:-1]
    def getOffsetAtEnd(self):
        return self.getOffset()[1:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _geometry = None
    def getGeometry(self):
        r = self._geometry
        if r is None:
            r = self.font.geometry[self._xidx]
            self._geometry = r
        return r
    geometry = property(getGeometry)


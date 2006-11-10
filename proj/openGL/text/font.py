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
from TG.openGL.text import fontData
from TG.openGL.text import fontTexture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Font(object):
    LayoutAlgorithm = blockMosaic.BlockMosaicAlg
    FontTexture = fontTexture.FontTexture
    FontGeometryArray = fontData.FontGeometryArray
    FontAdvanceArray = fontData.FontAdvanceArray
    FontTextDataFactory = staticmethod(fontData.FontTextData.factoryFor)
    FontTextData = None

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

    def textData(self, text=''):
        return self.FontTextData(text)

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

        gidxMap = {}
        aidxCounter = itertools.count(1)
        for char, gidx in face.iterCharIndexes(charset, True):
            aidx = aidxCounter.next()
            charMap.setdefault(char, aidx)
            gidxMap.setdefault(gidx, aidx)
        count = aidxCounter.next()

        # create a texture for the font mosaic, and run the mosaic algorithm
        self._compileKerningMap(face, gidxMap)
        mosaic = self._compileTexture(face, gidxMap)
        self._compileData(face, count, gidxMap, mosaic)
        self._compileFontTextData()

    def _compileKerningMap(self, face, gidxMap):
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

    def _compileFontTextData(self):
        self.FontTextData = self.FontTextDataFactory(self)


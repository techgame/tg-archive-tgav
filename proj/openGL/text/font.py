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
    FontTextData = fontData.FontTextData

    texture = None

    charMap = None
    geometry = None
    advance = None
    kerningMap = None

    pointSize = (1./64., 1./64.)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, face=None, charset=NotImplemented, compile=True):
        if charset is not NotImplemented:
            self.setCharset(charset, False)

        if face:
            self.setFace(face, compile)

    def __repr__(self):
        klass = self.__class__
        fmt = '<%s.%s %%s>' % (klass.__module__, klass.__name__,)
        info = '"%(familyName)s(%(styleName)s):%(lastSize)s"' % self.face.getInfo()
        return fmt % (info,)

    @classmethod
    def fromFilename(klass, filename, size=None, dpi=None, charset=None):
        self = klass()
        self.loadFace(filename, size, dpi, charset)
        return self

    def loadFace(self, filename, size=None, dpi=None, charset=NotImplemented):
        face = FreetypeFace(filename)
        if size is not None:
            face.setSize(size, dpi)
        self.setFace(face)

        if charset is not NotImplemented:
            self.setCharset(charset)
        self._markDirty()

    _face = None
    def getFace(self):
        return self._face
    def setFace(self, face):
        self._face = face
        self._markDirty()
    face = property(getFace, setFace)

    def getSize(self):
        return self.face.getSize()
    def setSize(self, size, dpi=None):
        face = self.face
        if size is not None:
            face.setSize(size, dpi)
        self._markDirty()
    size = property(getSize, setSize)

    _charset = None
    def getCharset(self):
        return self._charset
    def setCharset(self, charset):
        self._charset = charset
        self._markDirty()
    charset = property(getCharset, setCharset)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Text translation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def textData(self, text=''):
        self.compileIfDirty()
        return self.FontTextData(text)

    def translate(self, text):
        self.compileIfDirty()
        return map(self.charMap.get, text)

    def kernIndexes(self, idx, default=[0., 0., 0.]):
        km = self.kerningMap
        if not km or len(idx) < 2:
            return None
        self.compileIfDirty()
        
        r = asarray([km.get(e, default) for e in zip(idx, idx[1:])], float32)
        return r.reshape((-1, 1, 3))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Font compilation
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _markDirty(self, dirty=True):
        self._dirty = dirty

    def compileIfDirty(self):
        if self._dirty: self.recompile()

    def recompile(self):
        return self.compile(self.face, self.charset)

    def compile(self, face, charset):
        self.setFace(face)
        self.setCharset(charset)

        self.lineAdvance = self.FontAdvanceArray.fromItem((0., face.lineHeight*self.pointSize[1], 0.))

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

        self._dirty = False

    def _compileKerningMap(self, face, gidxMap):
        if not face.hasFlag('kerning'):
            return

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
        self.texture = texture

        mosaic, mosaicSize = self._compileGlyphMosaic(face, gidxMap, texture.getMaxTextureSize())
        texture.createMosaic(mosaicSize)
        return mosaic

    def _compileGlyphMosaic(self, face, gidxMap, maxSize):
        alg = self.LayoutAlgorithm((maxSize, maxSize))

        mosaic = {}
        for gidx in gidxMap.iterkeys():
            glyph = face.loadGlyph(gidx)
            glyph.render()
            size = glyph.bitmapSize
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
            self.texture.select()
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
            advance[aidx] = advanceFrom(glyph.advance, pointSize)

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
        return (advance[0]*ptw, advance[1]*pth, 0.)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _compileFontTextData(self):
        self.FontTextData = self.FontTextData.factoryUpdateFor(self)


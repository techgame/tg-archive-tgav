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

from itertools import izip

import numpy

from TG.freetype2.face import FreetypeFace

from TG.openGL import fontTexture
from TG.openGL import fontGeometry

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Font(object):
    TextureFactory = fontTexture.FontTextureRect

    pointSize = (1./64., 1./64.)

    def __init__(self, ftFace=None):
        if ftFace is not None:
            self.setFace(ftFace)

    def getFace(self):
        return self._ftFace
    def setFace(self, ftFace):
        self._ftFace = ftFace
    face = property(getFace, setFace)

    @classmethod
    def fromFilename(klass, filename, size, dpi=None):
        ftFace = FreetypeFace(filename)
        ftFace.setSize(size, dpi)
        return klass(ftFace)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _texture = None
    def getTexture(self):
        texture = self._texture
        if texture is None:
            texture = self.createTexture()
            self._texture = texture
        return texture
    def setTexture(self, texture):
        if texture is self._texture:
            return
        self._texture = texture
        self._invalidate()
    texture = property(getTexture, setTexture)

    def createTexture(self):
        return self.TextureFactory()

    _charset = None
    def getCharset(self):
        result = self._charset
        if result is None:
            return u''.join(c for c,i in self.face.iterAllChars())
        return result
    def setCharset(self, charset):
        if charset is self._charset:
            return
        self._charset = charset
        self._invalidate()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _invalidate(self):
        pass

    def configure(self):
        face = self.face
        charset = self.getCharset()

        texCoordMapping = self.texture.renderFace(face, face.iterUniqueCharIndexes(charset))
        texCoordMapping.sort()

        count = len(texCoordMapping)+1

        glyphToIndexMap = dict((glyphIndex, arrIndex) for arrIndex, (glyphIndex, texCoords) in enumerate(texCoordMapping))

        lookup=glyphToIndexMap.get
        idxOf=face.getOrdinalIndex
        charToIndexMap = numpy.empty(65535, dtype=numpy.uint16)
        for i in xrange(len(charToIndexMap)):
            charToIndexMap[i] = lookup(idxOf(i), 0)
        charToIndexMap[0] = count-1

        geometry = fontGeometry.FontGeometryArray.fromFormat((count, 4), dataFormat='t2f,v3f')
        advance = numpy.zeros((count, 4, 3), numpy.float32)

        loadGlyph = face.loadGlyph
        pointSize = self.pointSize
        verticesFrom = self._verticesFrom
        advanceFrom = self._advanceFrom
        for geoEntry, advEntry, (glyphIndex, texCoords) in izip(geometry, advance, texCoordMapping):
            glyph = loadGlyph(glyphIndex)

            geoEntry['t'] = texCoords
            geoEntry['v'] = verticesFrom(glyph.metrics, pointSize)

            advEntry[:] = advanceFrom(glyph.advance, pointSize)

        geoEntry = geometry[-1]
        geoEntry['t'] = [(0., 0.)]*4
        geoEntry['v'] = [(0., 0., 0.)]*4
        advance[-1] = [(0., 0., 0.)]*4

        #self.glyphToIndexMap = glyphToIndexMap
        self._indexMap = charToIndexMap
        self._geometry = geometry
        self._advance = advance

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _verticesFrom(self, metrics, (ptw, pth)):
        x0 = (metrics.horiBearingX) * ptw
        y0 = (metrics.horiBearingY - metrics.height) * pth

        x1 = (metrics.horiBearingX + metrics.width) * ptw
        y1 = (metrics.horiBearingY) * pth

        return [(x0, y0, 0.), (x1, y0, 0.), (x1, y1, 0.), (x0, y1, 0.)]

    def _advanceFrom(self, advance, (ptw, pth)):
        return [(advance[0]*ptw, advance[1]*pth, 0.)]*4

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getGeoAndAdv(self, text):
        text = [0] + map(ord, text.encode('utf-8'))
        idx = self._indexMap[text]
        adv = self._advance[idx]
        geo = self._geometry[idx[1:]]
        geo.dataFormat = self._geometry.dataFormat
        return idx, geo, adv

    def layout(self, text):
        idx, geo, adv = self.getGeoAndAdv(text)
        advSum = adv.cumsum(0)
        geo['v'] += advSum[:-1]

        def textRenderObj(tex=self.texture):
            tex.select()
            geo.draw(gl.GL_QUADS)
            tex.deselect()
        return geo, advSum[-1][-1], textRenderObj

    def render(self, text):
        textRenderObj = self.layout(text)[-1]
        textRenderObj()


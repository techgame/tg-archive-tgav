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
from face import FreetypeFace

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def asFreq(d):
    freq = dict()
    for e in d: 
        freq[e] = freq.get(e, 0) + 1
    return ', '.join('%s:%s' % e for e in freq.iteritems())

class FreetypeFontFace(object):
    FreetypeFaceFactory = FreetypeFace

    def __init__(self, filename, fontSize=None):
        self.setFilename(filename)
        if fontSize:
            self.setFontSize(fontSize)

    _filename = ''
    def getFilename(self):
        return self._filename
    def setFilename(self, filename):
        self._filename = filename
        self.loadFaceFromFilename(filename)
    filename = property(getFilename, setFilename)

    def loadFaceFromFilename(self, filename):
        face = self.FreetypeFaceFactory(filename)
        self.setFace(face)

    face = None
    def getFace(self):
        return self.face
    def setFace(self, face):
        if self.face is not None:
            self._delFace(self.face)
        self.face = face
        self._initFace(self.face)

    def _initFace(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def _delFace(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    fontSize = 16
    def getFontSize(self):
        return self.fontSize
    def setFontSize(self, fontSize):
        self.fontSize = fontSize
        self._setFaceSize(fontSize)
    def _setFaceSize(self, fontSize):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def iterKerning(self, chars, floating=True):
        face = self.face
        if floating:
            ptDiv = float(1<<6)
            l = chars[0]
            for r in chars[1:]:
                kh, kv = face.getKerning(l, r)
                yield (kh/ptDiv, kv/ptDiv)
                l = r
        else:
            l = chars[0]
            for r in chars[1:]:
                kh, kv = face.getKerning(l, r)
                yield (kh>>6, kv>>6)
                l = r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def printInfo(self):
        ptDiv = float(1<<6)
        face = self.face
        print 'names:'
        print '    file:', self.filename
        print '    postscript:', face.getPostscriptName()
        print '    family:', face.familyName
        print '    style:', face.styleName
        print

        print 'face flags:', hex(face.faceFlags), ', '.join(face.faceFlagsList)
        print 'number faces:', face.numFaces, '(%s)' % (face.faceIndex,)
        print 'number glyphs:', face.numGlyphs
        print

        cm = face.charmap[0]
        print 'charmap encoding:', cm.encoding.value, 'index:', face.getCharmapIndex(face.charmap), 'plat_id:', cm.platform_id, 'encoding_id:',cm.encoding_id
        print 'number charmaps:', face.numCharmaps
        for cm in face.charmaps[:face.numCharmaps]:
            cm = cm[0]
            print '    encoding:', cm.encoding.value, 'plat_id:', cm.platform_id, 'encoding_id:',cm.encoding_id
        print

        print 'metrics:'
        print '    units per em:', face.unitsPerEM
        print '    ascender:', face.ascender / ptDiv, 'descender:', face.descender / ptDiv, 'height:', face.height / ptDiv
        print '    bbox:', [(face.bbox.xMin/ptDiv, face.bbox.xMax/ptDiv), (face.bbox.yMin/ptDiv, face.bbox.yMax/ptDiv)]
        print '    underline pos:', face.underlinePosition/ptDiv, 'thickness:', face.underlineThickness/ptDiv
        print '    max advance width:', face.maxAdvanceWidth/ptDiv, 'height:', face.maxAdvanceHeight/ptDiv
        print
        #metrics = self.face.size[0].metrics
        #maxFontHeight = (metrics.ascender - metrics.descender) >> 6
        #maxFontWidth = (metrics.max_advance)>>6

    def printGlyphStats(self, chars):
        horiAdv = []
        vertAdv = []
        widths = []
        heights = []
    
        ptDiv = float(1<<6)
        ptDiv16 = float(1<<16)
        face = self.face
        for char, glyph in face.iterGlyphs(chars):
            print 'char:', repr(char), 'name:', repr(face.getGlyphName(char))
            if glyph.numSubglyphs:
                print '    subglyphs:', glyph.numSubglyphs
            print '    advance:', (glyph.advance[0]>>6, glyph.advance[1]>>6), 'linear:', (glyph.linearHoriAdvance/ptDiv16, glyph.linearVertAdvance/ptDiv16)
            print '    (x, y), (w, h):', (glyph.bitmapLeft, glyph.bitmapTop), (glyph.bitmap.width, glyph.bitmap.rows)

            metrics = glyph.metrics
            print '    metrics:', (metrics.width/ptDiv, metrics.height/ptDiv)
            print '        hori:', (metrics.horiBearingX/ptDiv, metrics.horiBearingY/ptDiv, metrics.horiAdvance/ptDiv)
            print '        vert:', (metrics.vertBearingX/ptDiv, metrics.vertBearingY/ptDiv, metrics.vertAdvance/ptDiv)

            assert glyph.bitmap.num_grays == 256, bitmap.num_grays

            horiAdv.append(glyph.advance[0]>>6)
            vertAdv.append(glyph.advance[1]>>6)
            widths.append(glyph.bitmap.width)
            heights.append(glyph.bitmap.rows)
            print

        print
        print 'widths: ', sum(widths), (min(widths), max(widths)), asFreq(widths)
        print 'heights:', sum(heights), (min(heights), max(heights)), asFreq(heights)
        print 'horiAdv:', sum(horiAdv), (min(horiAdv), max(horiAdv)), asFreq(horiAdv)
        print 'vertAdv:', sum(vertAdv), (min(vertAdv), max(vertAdv)), asFreq(vertAdv)
        print

    def printKerning(self, chars):
        for k in self.iterKerning(chars):
            print k


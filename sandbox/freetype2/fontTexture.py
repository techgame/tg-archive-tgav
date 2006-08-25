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

from itertools import izip, groupby
import string
from bisect import bisect_left, bisect_right

from ctypes import byref, c_void_p

from TG.freetype2.fontFace import FreetypeFontFace

from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

from TG.openGL.texture import Texture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFace(FreetypeFontFace):
    def _initFace(self, face):
        self.setFontSize(self.getFontSize())

    def _delFace(self, face):
        print 'Should deallocate face'

    def _setFaceSize(self, fontSize):
        self.face.setPixelSize(fontSize)

    def load(self, width=256, height=256):
        self.width = width
        self.height = height
        self.texture = Texture(GL_TEXTURE_2D, GL_INTENSITY,
                wrap=gl.GL_CLAMP, genMipmaps=True,
                magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR_MIPMAP_LINEAR)

        self.data = self.texture.data2d(size=(width, height), format=GL_LUMINANCE, dataType=GL_UNSIGNED_BYTE)
        self.data.texBlank()
        self.data.setImageOn(self.texture)
        self.data.texClear()

        return self.texture

    def loadChars(self, chars):
        data = self.data
        texture = self.texture
        width = self.width
        height = self.height
        x = 0; y = 0
        pixelStore = self.data.newPixelStore(alignment=1, rowLength=0)
        maxRowHeight = 0
        for char, glyph in self.face.iterChars(chars):
            bitmap = glyph.bitmap
            assert bitmap.num_grays == 256, bitmap.num_grays
            (w,h) = bitmap.width, bitmap.rows

            if (x+w) > width:
                x = 0
                y += maxRowHeight + 0
                maxRowHeight = h
            else:
                maxRowHeight = max(h, maxRowHeight)

            pixelStore.rowLength = bitmap.pitch
            data.posSize = (x,y), (w,h)
            data.texCData(bitmap.buffer)

            data.setSubImageOn(texture)
            x += w + 0

        data.texClear()
        return texture

    def loadSizes(self, chars):
        iGlyphs = self.face.iterGlyphs(chars)
        return dict((c, g.bitmapSize) for c, g in iGlyphs)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def groupByDelta(v):
    dv = [e1-e0 for e0, e1 in zip(v[:-1], v[1:])] + [0]
    dAvg = sum(dv, 0.)/len(dv)

    p = v[0]
    r = [p]
    for n in heights[1:]:
        if n-p > dAvg:
            yield r
            r = []
        r.append(n)
        p = n

    if r: yield r

if __name__=='__main__':
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',

            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT'
            }
    fft = GLFreetypeFace(fonts['LucidaGrande'], 128)
    #fft.printInfo()
    #fft.printGlyphStats('AV')
    #fft.printKerning('fi')

    twos = [1<<s for s in xrange(31)]
    nextPower2 = lambda v: twos[bisect_right(twos, v)]

    maxTextureWidth = 2048
    maxTextureHeight = 2048
    maxTextureArea = maxTextureWidth * maxTextureHeight

    sizes = fft.loadSizes(string.uppercase + string.lowercase)
    area = sum((2+e[0])*(2+e[1]) for e in sizes.itervalues())
    wMin = min(e[0] for e in sizes.itervalues())
    hMin = min(e[1] for e in sizes.itervalues())
    wMax = max(e[0] for e in sizes.itervalues())
    hMax = max(e[1] for e in sizes.itervalues())

    areaP2, wMaxp2, hMaxp2 = map(nextPower2, (area, wMax, hMax))
    assert areaP2 <= maxTextureArea

    heightMap = {}
    for c, (w, h) in sizes.iteritems():
        heightMap.setdefault(h, []).append((w, c))

    heights = sorted(heightMap.keys())

    gHeights = [list(g) for g in groupByDelta(heights)]

    hwSumMap = {}
    widthsP2 = []
    for h, wcm in heightMap.iteritems():
        wcm.sort()
        w = sum(e[0] for e in wcm)
        wp2 = nextPower2(w)
        widthsP2.append(wp2)
        hwSumMap[h] = (wp2, w, wp2-w)

    avgWidth = sum(widthsP2) / len(widthsP2)
    avgWidthP2 = nextPower2(avgWidth)

    def layout(totalW, totalH):
        hMax = 0
        waste = 0
        x = y = 0
        for hGrp in gHeights:
            print
            print 'HGrp:', hGrp
            for h in hGrp:
                #print '  Height:', h, 'y:', y
                widthMap = heightMap[h]

                if not widthMap:
                    continue

                if h > hMax:
                    heightWaste = (h-hMax)*x

                    endWaste = (totalW-x) * hMax
                    if endWaste < heightWaste:
                        print 'We should make a new row instead:', (endWaste, heightWaste)

                    waste += heightWaste
                    hMax = h
                else: heightWaste = 0

                elemH = h
                while widthMap:
                    widthRemain = totalW-x
                    i = bisect_right(widthMap, (widthRemain,'\xffff')) - 1
                    if i >= 0:
                        elemW, elemC = widthMap.pop(i)
                        yield (elemC, (x, y), (elemW, elemH))

                        # advance to next available
                        x += elemW + 2
                    else:
                        endWaste = hMax*widthRemain
                        waste += endWaste
                        #print '    row:', (x, totalW)
                        print '    waste:', (heightWaste, endWaste, (widthRemain, hMax))
                        #print '      widths:', widthRemain, [e[0] for e in widthMap]
                        #print
                        x = 0
                        y += hMax + 2
                        hMax = h
                        #print '  Height:', h, 'y:', y

        print 'Remaining Blocks:'
        print '    Last Row:', (totalW-x, hMax, (totalW-x)*hMax)
        print '    Bottom:', (totalW, totalH-(y+hMax), totalW*(totalH-(y+hMax)))


    if 1:
        print area, (wMin, wMax), (hMin, hMax)
        print areaP2, wMaxp2, hMaxp2
        print 'Area:', area, 'areaP2:', areaP2, 'max:', maxTextureArea, 'ratio: 1 /', maxTextureArea/areaP2
        print 'Avg Width:', avgWidth, nextPower2(avgWidth), areaP2/avgWidthP2

    if 1:
        list(layout(avgWidthP2, areaP2/avgWidthP2))

    if 0:
        print 
        for h, (widthP2, width, unused) in sorted(hwSumMap.items()):
            print '    h: %6s wp2: %6s width: %6s unused: %6s' % (h, widthP2, width, unused)
        print
        print


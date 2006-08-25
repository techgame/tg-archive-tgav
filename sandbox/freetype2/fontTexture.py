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

if __name__=='__main__':
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',

            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT'
            }
    fft = GLFreetypeFace(fonts['LucidaGrande'], 64)
    #fft.printInfo()
    #fft.printGlyphStats('AV')
    #fft.printKerning('fi')

    from blockMosaicLayout import BlockMosaicAlg

    alg = BlockMosaicAlg()
    alg.maxSize = (2048, 2048)

    sizes = fft.loadSizes(string.uppercase + string.lowercase)
    for char, bmSize in sizes.iteritems():
        alg.addBlock(bmSize, char)

    for e in alg.layout():
        if 0: 
            print e.key, e.pos, e.size


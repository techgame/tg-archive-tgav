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

    def loadChars(self, chars):
        size, charLayout = self.layoutMosaic(chars)
        texture = self._newTexure(size)
        self._renderCharsToTexture(charLayout, texture)
        return texture

    def _newTexure(self, size):
        #working: texture: GL_INTENSITY, texEnv: GL_MODULTE, glBlendFunc: (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        texture = Texture(GL_TEXTURE_2D, GL_INTENSITY,
                wrap=gl.GL_CLAMP, genMipmaps=True,
                magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR_MIPMAP_LINEAR)

        self.data = texture.data2d(size=size, format=GL_LUMINANCE, dataType=GL_UNSIGNED_BYTE)
        self.data.texBlank()
        self.data.setImageOn(texture)
        self.data.texClear()

        return texture

    def _renderCharsToTexture(self, charLayout, texture):
        data = self.data
        pixelStore = self.data.newPixelStore(alignment=1, rowLength=0)

        for char, pos in charLayout.iteritems():
            glyph = self.face.loadChar(char)
            pixelStore.rowLength = glyph.bitmap.pitch
            data.posSize = (pos, glyph.bitmapSize)
            data.texCData(glyph.bitmap.buffer)
            data.setSubImageOn(texture)

        data.texClear()
        return texture

    def layoutMosaic(self, chars):
        from blockMosaicLayout import BlockMosaicAlg

        alg = BlockMosaicAlg()
        alg.maxSize = (2048, 2048)

        for char, glyph in self.face.iterGlyphs(chars):
            alg.addBlock(glyph.bitmapSize, key=char)

        rgn, iLayout = alg.layout()
        
        try:
            charLayout = dict((e.key, e.pos) for e in iLayout)
        except Exception:
            rgn.w <<= 1
            rgn, iLayout = alg.layoutSize(rgn.size)
            charLayout = dict((e.key, e.pos) for e in iLayout)
        #rgn.printUnused(('LastRow', 'Bottom'))

        return rgn.size, charLayout

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

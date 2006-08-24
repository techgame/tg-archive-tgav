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

from string import printable

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

    def load(self):
        fontSize = 64
        width, height = 256, 256

        texture = Texture(GL_TEXTURE_2D, GL_INTENSITY,
                wrap=gl.GL_CLAMP, magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR_MIPMAP_LINEAR, genMipmaps=True)

        data = texture.data2d(size=(width, height), format=GL_LUMINANCE, dataType=GL_UNSIGNED_BYTE)
        data.texBlank()
        texture.setImage(data)

        x = 0
        y = 0
        #glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        #glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)
        data.texClear()
        pixelStore = data.newPixelStore()
        pixelStore.alignment = 1
        pixelStore.rowLength = 0
        maxRowHeight = 0
        for char, glyph in self.face.iterChars('What fun!'):
            bitmap = glyph.bitmap
            assert bitmap.num_grays == 256, bitmap.num_grays
            (w,h) = bitmap.width, bitmap.rows
            maxRowHeight = max(h, maxRowHeight)

            if (x+w) > width:
                x = 0
                y += maxRowHeight + 0
                maxRowHeight = h

            pixelStore.rowLength = bitmap.pitch
            data.posSize = (x,y), (w,h)
            data.texCData(bitmap.buffer)

            data.select().next()
            texture.setSubImage(data)
            x += w + 0

        self.texture = texture
        return self.texture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT'
            }
    fft = FreetypeFontFaceTexture(fonts['Arial'])
    fft.printInfo()
    fft.printGlyphStats('AV')
    fft.printKerning('fi')


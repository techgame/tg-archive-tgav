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

from ctypes import byref, c_void_p, c_uint

from TG.freetype2.fontFace import FreetypeFontFace

from TG.openGL.texture import Texture
from TG.openGL.raw import gl, glu, glext

from blockMosaicLayout import BlockMosaicAlg

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFaceBasic(FreetypeFontFace):
    bUseMipmaps = False
    texTarget = None #gl.GL_TEXTURE_2D or glext.GL_TEXTURE_RECTANGLE_ARB
    texFormat = gl.GL_INTENSITY

    def _initFace(self, face):
        self.setFontSize(self.getFontSize())

    def _delFace(self, face):
        print 'Should deallocate face'

    def _setFaceSize(self, fontSize):
        self.face.setPixelSize(fontSize)

    def loadChars(self, chars):
        self.texture = self._createTexure()

        usedSize, charLayout = self.layoutMosaic(chars)
        self._layoutMap = dict((b.key, b) for b in charLayout)

        self._clearTexureData(self.texture, usedSize)
        self._renderCharsToTexture(charLayout, self.texture)

        return self.texture

    def _createTexure(self):
        #working: texture: GL_INTENSITY, texEnv: GL_MODULTE, gl.glBlendFunc: (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        texture = Texture(self.texTarget, self.texFormat, wrap=gl.GL_CLAMP)
        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))

        if self.bUseMipmaps:
            texture.set(genMipmaps=True, magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR_MIPMAP_LINEAR)
        else:
            texture.set(genMipmaps=False, magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR)

        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))
        return texture

    def _clearTexureData(self, texture, size):
        self.size = size

        self.data = texture.data2d(size=size, format=gl.GL_LUMINANCE, dataType=gl.GL_UNSIGNED_BYTE)
        self.data.texBlank()
        self.data.setImageOn(texture)
        self.data.texClear()
        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))

        return texture

    def _renderCharsToTexture(self, charLayout, texture):
        data = self.data
        pixelStore = self.data.newPixelStore(alignment=1, rowLength=0)

        mapping = {}

        listBase = gl.glGenLists(len(charLayout))
        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))
        glLists = range(listBase, listBase + len(charLayout) + 1)

        for block in charLayout:
            glyphIndex = block.key

            glyph = self.face.loadGlyph(glyphIndex)
            glyph.render()
            pixelStore.rowLength = glyph.bitmap.pitch
            data.posSize = (block.pos, block.size)
            data.texCData(glyph.bitmap.buffer)
            data.setSubImageOn(texture)

            # compile the glyph to an openGL list so everything is stored on the graphics card
            glListIdx = glLists.pop(0)
            self._compileGlyph(glListIdx, glyph, block.pos, block.size)
            if gl.glIsList(glListIdx):
                mapping[glyphIndex] = glListIdx

        data.texClear()
        self._indexGLListMap = mapping
        return texture

    def layoutMosaic(self, chars):
        alg = BlockMosaicAlg()
        if 1:
            alg.maxWidth = Texture.getMaxTextureSizeFor(self.texTarget)
        else:
            alg.maxWidth = 512

        chars = u'\x00 \t\n\r' + chars
        self.charMap = self.face.charIndexMap(chars)

        for glyphIndex, glyph in self.face.iterUniqueGlyphs(chars):
            alg.addBlock(glyph.bitmapSize, key=glyphIndex)

        usedSize, glyphLayout, unplaced = alg.layout()
        assert len(unplaced) == 0, unplaced

        return usedSize, glyphLayout

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def drawFontTexture(self):
        for e in self.texture.select():
            self._drawTextureRect()

    def _drawTextureRect(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def drawString(self, chars):
        glListIds = [self._indexGLListMap.get(i) for i in self.face.iterCharIndexes(chars)]
        glListIds = (c_uint*len(glListIds))(*glListIds)
        for e in self.texture.select():
            gl.glCallLists(len(glListIds), gl.GL_UNSIGNED_INT, glListIds)

    def drawChar(self, char):
        glyph = self.face.loadGlyph(char)
        block = self._layoutMap[glyph.index]

        for e in self.texture.select():
            self._drawGlyphRect(glyph, block.pos, block.size)

    def _compileGlyph(self, glListId, glyph, pos, size):
        gl.glNewList(glListId, gl.GL_COMPILE)
        self._drawGlyphRect(glyph, pos, size)
        gl.glEndList()

    def _drawGlyphRect(self, glyph, pos, size):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFace2D(GLFreetypeFaceBasic):
    texTarget = gl.GL_TEXTURE_2D

    def _clearTexureData(self, texture, size):
        size = map(Texture.nextPowerOf2, size)
        GLFreetypeFaceBasic._clearTexureData(self, texture, size)

    def _drawTextureRect(self):
        width, height = self.size

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(0, 0)
        gl.glVertex2f(0, height)

        gl.glTexCoord2s(0, 1)
        gl.glVertex2f(0, 0)

        gl.glTexCoord2s(1, 1)
        gl.glVertex2f(width, 0)

        gl.glTexCoord2s(1, 0)
        gl.glVertex2f(width, height)
        gl.glEnd()

    def _drawGlyphRect(self, glyph, pos, size):
        texW, texH = map(float, self.size)
        tx0 = pos[0]/texW
        ty0 = pos[1]/texH
        tx1 = (pos[0] + size[0])/texW
        ty1 = (pos[1] + size[1])/texH

        width, height = size

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(tx0, ty0)
        gl.glVertex2f(0, height)

        gl.glTexCoord2f(tx0, ty1)
        gl.glVertex2f(0, 0)

        gl.glTexCoord2f(tx1, ty1)
        gl.glVertex2f(width, 0)

        gl.glTexCoord2f(tx1, ty0)
        gl.glVertex2f(width, height)
        gl.glEnd()

        gl.glTranslatef(width, 0, 0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFaceRect(GLFreetypeFaceBasic):
    texTarget = glext.GL_TEXTURE_RECTANGLE_ARB

    def _drawTextureRect(self):
        width, height = self.size

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(0, 0)
        gl.glVertex2s(0, height)

        gl.glTexCoord2s(0, height)
        gl.glVertex2s(0, 0)

        gl.glTexCoord2s(width, height)
        gl.glVertex2s(width, 0)

        gl.glTexCoord2s(width, 0)
        gl.glVertex2s(width, height)
        gl.glEnd()

    def _drawGlyphRect(self, glyph, pos, size):
        tx0 = pos[0]
        ty0 = pos[1]
        tx1 = tx0 + size[0]
        ty1 = ty0 + size[1]

        width, height = size

        x0 = glyph.bitmapLeft
        x1 = x0 + width
        y0 = glyph.bitmapTop - height
        y1 = y0 + height

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(tx0, ty0)
        gl.glVertex2s(x0, y1)

        gl.glTexCoord2s(tx0, ty1)
        gl.glVertex2s(x0, y0)

        gl.glTexCoord2s(tx1, ty1)
        gl.glVertex2s(x1, y0)

        gl.glTexCoord2s(tx1, ty0)
        gl.glVertex2s(x1, y1)
        gl.glEnd()

        ax, ay =  glyph.advance
        gl.glTranslatef(ax/64., ay/64., 0)


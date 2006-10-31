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

from TG.openGL.blockMosaic import BlockMosaicAlg

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFaceBasic(FreetypeFontFace):
    bUseMipmaps = False
    texTarget = None #gl.GL_TEXTURE_2D or glext.GL_TEXTURE_RECTANGLE_ARB
    texFormat = gl.GL_ALPHA
    dataFormat = gl.GL_ALPHA

    pointSize=1./64.

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
        texture = Texture(self.texTarget, format=self.texFormat, wrap=gl.GL_CLAMP)
        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))

        if self.bUseMipmaps:
            texture.set(genMipmaps=True, magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR_MIPMAP_LINEAR)
        else:
            texture.set(genMipmaps=False, magFilter=gl.GL_LINEAR, minFilter=gl.GL_LINEAR)

        if gl.glGetError(): raise RuntimeError("GL: %s" % (gl.glGetError(),))
        return texture

    def _clearTexureData(self, texture, size):
        self.size = size

        self.data = texture.data2d(size=size, format=self.dataFormat, dataType=gl.GL_UNSIGNED_BYTE)
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
            self._compileGlyph(glListIdx, glyph, block)
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
        self.texture.select()
        self._drawTextureRect()

    def _drawTextureRect(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    @classmethod
    def selectScale(klass):
        gl.glPushMatrix()
        gl.glScalef(klass.pointSize, klass.pointSize, 1)
        yield
        gl.glPopMatrix()

    def select(self):
        self.texture.select()
        gl.glPushMatrix()
        yield self
        gl.glPopMatrix()

    def drawBreak(self, advance=True):
        gl.glPopMatrix()
        vAdvance = self.face.size[0].metrics.height
        #vAdvance =  self.face.ascender - self.face.descender
        #vAdvance =  self.face.bbox.yMax - self.face.bbox.yMin
        #print 'advance:', vAdvance/64., self.face.bbox.xMin/64., self.face.bbox.xMax/64., self.face.bbox.yMin/64., self.face.bbox.yMax/64.
        gl.glTranslatef(0, -vAdvance, 0)
        gl.glPushMatrix()

    def drawString(self, chars):
        glListIds = [self._indexGLListMap.get(i) for i in self.face.iterCharIndexes(chars)]
        glListIds = (c_uint*len(glListIds))(*glListIds)
        gl.glCallLists(len(glListIds), gl.GL_UNSIGNED_INT, glListIds)

    def drawStringSlow(self, chars, fromRight=False):
        if fromRight:
            chars = chars[::-1]
        loadGlyph = self.face.loadGlyph
        layoutMap = self._layoutMap

        for glyphIndex in self.face.iterCharIndexes(chars):
            self._drawGlyphRect(loadGlyph(glyphIndex), layoutMap[glyphIndex], fromRight)

    def _compileGlyph(self, glListId, glyph, block):
        gl.glNewList(glListId, gl.GL_COMPILE)
        self._drawGlyphRect(glyph, block)
        gl.glEndList()

    def _drawGlyphRect(self, glyph, block):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFace2D(GLFreetypeFaceBasic):
    texTarget = gl.GL_TEXTURE_2D

    def _clearTexureData(self, texture, size):
        size = map(Texture.nextPowerOf2, size)
        GLFreetypeFaceBasic._clearTexureData(self, texture, size)

    def _drawTextureRect(self):
        texW, texH = self.size

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(0, 0)
        gl.glVertex2s(0, texH)

        gl.glTexCoord2s(0, 1)
        gl.glVertex2s(0, 0)

        gl.glTexCoord2s(1, 1)
        gl.glVertex2s(texW, 0)

        gl.glTexCoord2s(1, 0)
        gl.glVertex2s(texW, texH)
        gl.glEnd()

    def _drawGlyphRect(self, glyph, block, fromRight=False):
        m = glyph.metrics
        x0 = m.horiBearingX
        x1 = x0 + m.width
        y1 = m.horiBearingY
        y0 = y1 - m.height
        ax, ay = glyph.advance

        texW, texH = map(float, self.size)
        tx0 = block.x/texW
        ty0 = block.y/texH
        tx1 = block.x1/texW
        ty1 = block.y1/texH

        if fromRight:
            gl.glTranslatef(-ax, -ay, 0)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(tx0, ty0)
        gl.glVertex2f(x0, y1)

        gl.glTexCoord2f(tx0, ty1)
        gl.glVertex2f(x0, y0)

        gl.glTexCoord2f(tx1, ty1)
        gl.glVertex2f(x1, y0)

        gl.glTexCoord2f(tx1, ty0)
        gl.glVertex2f(x1, y1)
        gl.glEnd()

        if not fromRight:
            gl.glTranslatef(ax, ay, 0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLFreetypeFaceRect(GLFreetypeFaceBasic):
    texTarget = glext.GL_TEXTURE_RECTANGLE_ARB

    def _drawTextureRect(self):
        texW, texH = self.size

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(0, 0)
        gl.glVertex2s(0, texH)

        gl.glTexCoord2s(0, texH)
        gl.glVertex2s(0, 0)

        gl.glTexCoord2s(texW, texH)
        gl.glVertex2s(texW, 0)

        gl.glTexCoord2s(texW, 0)
        gl.glVertex2s(texW, texH)
        gl.glEnd()

    def _drawGlyphRect(self, glyph, block, fromRight=False):
        m = glyph.metrics
        x0 = m.horiBearingX
        x1 = x0 + m.width
        y1 = m.horiBearingY
        y0 = y1 - m.height
        ax, ay = glyph.advance

        tx0 = block.x
        ty0 = block.y
        tx1 = block.x1
        ty1 = block.y1

        if fromRight:
            gl.glTranslatef(-ax, -ay, 0)

        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2s(tx0, ty0)
        gl.glVertex2f(x0, y1)

        gl.glTexCoord2s(tx0, ty1)
        gl.glVertex2f(x0, y0)

        gl.glTexCoord2s(tx1, ty1)
        gl.glVertex2f(x1, y0)

        gl.glTexCoord2s(tx1, ty0)
        gl.glVertex2f(x1, y1)
        gl.glEnd()

        if not fromRight:
            gl.glTranslatef(ax, ay, 0)


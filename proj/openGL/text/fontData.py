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

import weakref
from numpy import ndarray, float32, asarray

from TG.openGL.data.vertexArrays import VertexArray
from TG.openGL.data.interleavedArrays import InterleavedArray

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontGeometryArray(InterleavedArray):
    drawMode = gl.GL_QUADS
    gldtype = InterleavedArray.gldtype.copy()
    gldtype.setDefaultFormat(gl.GL_T2F_V3F)

class FontAdvanceArray(VertexArray):
    gldtype = VertexArray.gldtype.copy()
    gldtype.setDefaultFormat('3f')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextData(object):
    font = None
    GeometryArray = FontGeometryArray
    AdvanceArray = FontAdvanceArray

    def setupFont(self, font):
        self.font = font
        if not isinstance(self, type):
            self.recompile()
    setupClassFont = classmethod(setupFont)

    @classmethod
    def factoryUpdateFor(klass, font):
        if font is klass.font:
            klass.setupClassFont(font)
            return klass
        else:
            subklass = type(klass)(klass.__name__+'_T_', (klass,), {})
            subklass.setupClassFont(font)
            return subklass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, text, font=None):
        if font is not None:
            self.setupFont(font)
        self.text = text

    _text = ""
    def getText(self):
        return self._text
    def setText(self, text):
        if text != self._text:
            self._text = text
            self.recompile()
    text = property(getText, setText)

    def __nonzero__(self):
        return bool(self.text)

    def getTexture(self):
        return self.font.texture
    texture = property(getTexture)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def recompile(self):
        if self.font is not None:
            self._xidx = self.font.translate(self.text or '')
        else:
            self._xidx = None
        self._advance = None
        self._lineAdvance = None
        self._geometry = None
        self._offset = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _lineAdvance = None
    def getLineAdvance(self):
        r = self._lineAdvance
        if r is None:
            r = self.font.lineAdvance
            self._lineAdvance = r
        return r
    lineAdvance = property(getLineAdvance)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _advance = None
    def getAdvance(self):
        r = self._advance
        if r is None:
            r = self.font.advance[[0]+self._xidx]
            k = self.font.kernIndexes(self._xidx)
            if k is not None:
                r[1:-1] += k
            self._advance = r
        return r

    _offset = None
    def getOffset(self):
        r = self._offset
        if r is None:
            r = self.getAdvance().cumsum(0)
            self._offset = r

        return r

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _geometry = None
    def getGeometry(self):
        r = self._geometry
        if r is None:
            r = self.font.geometry[self._xidx]
            self._geometry = r
        return r
    geometry = property(getGeometry)


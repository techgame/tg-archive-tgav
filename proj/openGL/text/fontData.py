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

from numpy import ndarray, float32, asarray

from TG.openGL.data import interleavedArrays

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontGeometryArray(interleavedArrays.InterleavedArrays):
    drawMode = gl.GL_QUADS
    dataFormat = gl.GL_T2F_V3F
    @classmethod
    def fromCount(klass, count):
        return klass.fromFormat((count, 4), klass.dataFormat)

class FontAdvanceArray(ndarray):
    @classmethod
    def fromCount(klass, count, dtype=float32):
        return klass((count, 1, 3), dtype)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextData(object):
    font = None
    texture = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def factoryFor(klass, font):
        subklass = type(klass)(klass.__name__+'_T_', (klass,), {})
        subklass.setupClassFont(font)
        return subklass

    def setupFont(self, font):
        self.font = font
        self.texture = font.texture
        if not isinstance(self, type):
            self._recache()
    setupClassFont = classmethod(setupFont)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, text, font=None):
        if font is not None:
            self.setupFont(font)
        self.text = text

    _text = ""
    def getText(self):
        return self._text
    def setText(self, text):
        self._text = text
        self._recache()
    text = property(getText, setText)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _recache(self):
        if self.font is not None:
            self._xidx = self.font.translate(self.text)
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

    def getPreAdvance(self):
        return self.getAdvance()[:-1]
    def getPostAdvance(self):
        return self.getAdvance()[1:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _offset = None
    def getOffset(self):
        r = self._offset
        if r is None:
            r = self.getAdvance().cumsum(0)
            self._offset = r

        return r

    def getOffsetAtStart(self):
        return self.getOffset()[:-1]
    def getOffsetAtEnd(self):
        return self.getOffset()[1:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _geometry = None
    def getGeometry(self):
        r = self._geometry
        if r is None:
            r = self.font.geometry[self._xidx]
            self._geometry = r
        return r
    geometry = property(getGeometry)


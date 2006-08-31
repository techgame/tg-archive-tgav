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

from raw import freetype as FT
import ctypes
from ctypes import byref, cast, c_void_p

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ptDiv = float(1<<6)
ptDiv16 = float(1<<16)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeFaceGlyph(object):
    #~ FreeType API interation ~~~~~~~~~~~~~~~~~~~~~~~~~~
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_GlyphSlot
    index = -1

    def __init__(self, glyph, face):
        self._as_parameter_ = glyph
        self.face = face

    def __nonzero__(self):
        if self._as_parameter_:
            return bool(self.index)
        else: return False
    @property
    def _glyph(self):
        return self._as_parameter_[0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def metrics(self):
        return self._glyph.metrics # FT_Glyph_Metrics
    @property
    def linearHoriAdvance(self):
        return self._glyph.linearHoriAdvance # FT_Fixed
    @property
    def linearVertAdvance(self):
        return self._glyph.linearVertAdvance # FT_Fixed
    @property
    def advance(self):
        adv = self._glyph.advance # FT_Vector
        return (adv.x, adv.y)
    @property
    def format(self):
        return self._glyph.format # FT_Glyph_Format
    @property
    def bitmap(self):
        return self._glyph.bitmap # FT_Bitmap
    @property
    def bitmapLeft(self):
        return self._glyph.bitmap_left # FT_Int
    @property
    def bitmapTop(self):
        return self._glyph.bitmap_top # FT_Int
    @property
    def outline(self):
        return self._glyph.outline # FT_Outline
    @property
    def numSubglyphs(self):
        return self._glyph.num_subglyphs # FT_UInt
    @property
    def subglyphs(self):
        return self._glyph.subglyphs # FT_SubGlyph
    @property
    def controlData(self):
        return self._glyph.control_data # c_void_p
    @property
    def controlLen(self):
        return self._glyph.control_len # c_long
    @property
    def lsbDelta(self):
        return self._glyph.lsb_delta # FT_Pos
    @property
    def rsbDelta(self):
        return self._glyph.rsb_delta # FT_Pos

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def bitmapSize(self):
        bm = self.bitmap
        return (bm.width, bm.rows)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _renderMode = None
    def getRenderMode(self):
        if self._renderMode is None:
            return self.face.renderMode
        return self._renderMode
    def setRenderMode(self, renderMode):
        self._renderMode = renderMode
    renderMode = property(getRenderMode, setRenderMode)

    _ft_renderGlyph = FT.FT_Render_Glyph
    def render(self, renderMode=None):
        if renderMode is None: 
            renderMode = self.renderMode
        self._ft_renderGlyph(renderMode)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def printInfo(self, out=None):
        print 'glyphIndex:', self.index, 'name:', repr(self.face.getGlyphName(self.index))
        if self.numSubglyphs:
            print >> out, '    subglyphs:', self.numSubglyphs
        print >> out, '    advance:', (self.advance[0]>>6, self.advance[1]>>6), 'linear:', (self.linearHoriAdvance/ptDiv16, self.linearVertAdvance/ptDiv16)
        print '    (x, y), (w, h):', (self.bitmapLeft, self.bitmapTop), (self.bitmap.width, self.bitmap.rows)

        metrics = self.metrics
        print >> out, '    metrics:', (metrics.width/ptDiv, metrics.height/ptDiv)
        print >> out, '        hori:', (metrics.horiBearingX/ptDiv, metrics.horiBearingY/ptDiv, metrics.horiAdvance/ptDiv)
        print >> out, '        vert:', (metrics.vertBearingX/ptDiv, metrics.vertBearingY/ptDiv, metrics.vertAdvance/ptDiv)


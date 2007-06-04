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

import weakref

import numpy
from numpy import frombuffer, ndarray, zeros

import ctypes
from ctypes import byref, cast, c_void_p, string_at

from raw import freetype as FT
from raw import ftglyph
from raw.errors import FreetypeException

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ptDiv = float(1<<6)
ptDiv16 = float(1<<16)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeGlyphSlot(object):
    #~ FreeType API interation ~~~~~~~~~~~~~~~~~~~~~~~~~~
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_GlyphSlot
    index = 0

    def __init__(self, glyphslot, face):
        self._as_parameter_ = glyphslot
        self.face = face

    def __nonzero__(self):
        if self._as_parameter_:
            return bool(self.index)
        else: return False

    def __repr__(self):
        klass = self.__class__
        return '<%s %s:%r>' % (klass.__name__, self.index, self.name)

    @property
    def _glyphslot(self):
        return self._as_parameter_[0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def name(self):
        return self.face.getGlyphName(self.index)

    @property
    def metrics(self):
        return self._glyphslot.metrics # FT_Glyph_Metrics
    @property
    def linearHoriAdvance(self):
        return self._glyphslot.linearHoriAdvance # FT_Fixed
    @property
    def linearVertAdvance(self):
        return self._glyphslot.linearVertAdvance # FT_Fixed
    @property
    def advance(self):
        return frombuffer(self._glyphslot.advance, 'l')
    @property
    def format(self):
        return self._glyphslot.format # FT_Glyph_Format

    def isBitmapFormat(self):
        format = self.format
        return format.value == format.FT_GLYPH_FORMAT_BITMAP

    def getBitmapArray(self, gbmp=None):
        if gbmp is None:
            gbmp = self._glyphslot.bitmap

        if gbmp.buffer:
            if gbmp.pixel_mode == '\x02':
                # gray bytes
                bmpstr = string_at(gbmp.buffer, gbmp.rows*gbmp.width)
                return ndarray((gbmp.rows, gbmp.width), 'B', bmpstr)

            if gbmp.pixel_mode == '\x01':
                bmpstr = string_at(gbmp.buffer, gbmp.rows*gbmp.width)
                bits = ndarray((gbmp.rows, abs(gbmp.pitch)), 'B', bmpstr)

                result = zeros((gbmp.rows, gbmp.width), 'B')
                for i in xrange(gbmp.width):
                    w = (bits[:] >> (7-i)) & 1
                    result[:, i] = 255 * w[:, 0]
                return result

            raise NotImplementedError("Unimplemented font bitmap pixel mode")

    @property
    def bitmap(self):
        return self._glyphslot.bitmap # FT_Bitmap
    @property
    def bitmapLeft(self):
        return self._glyphslot.bitmap_left # FT_Int
    @property
    def bitmapTop(self):
        return self._glyphslot.bitmap_top # FT_Int
    @property
    def outline(self):
        return self._glyphslot.outline # FT_Outline
    @property
    def numSubglyphs(self):
        return self._glyphslot.num_subglyphs # FT_UInt
    @property
    def subglyphs(self):
        return self._glyphslot.subglyphs # FT_SubGlyph
    @property
    def controlData(self):
        return self._glyphslot.control_data # c_void_p
    @property
    def controlLen(self):
        return self._glyphslot.control_len # c_long
    @property
    def lsbDelta(self):
        return self._glyphslot.lsb_delta # FT_Pos
    @property
    def rsbDelta(self):
        return self._glyphslot.rsb_delta # FT_Pos

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def bitmapSize(self):
        bm = self.bitmap
        return (bm.width, bm.rows)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _renderMode = None
    def getRenderMode(self):
        mode = self._renderMode
        if mode is None:
            mode = self.face.glyphRenderMode
        return mode
    def setRenderMode(self, renderMode):
        self._renderMode = renderMode
    renderMode = property(getRenderMode, setRenderMode)

    origin = ftglyph.FT_Vector(0, 0)
    _ft_renderGlyph = FT.FT_Render_Glyph
    _ft_glyphToBitmap = staticmethod(ftglyph.FT_Glyph_To_Bitmap)
    def render(self, renderMode=None):
        if renderMode is None: 
            renderMode = self.renderMode
        self._ft_renderGlyph(renderMode)

        # Glyph to bitmap process -- seems to produce same results as renderGlyph
        ##glyph = self.glyph
        ##self._ft_glyphToBitmap(glyph, renderMode, self.origin, 1)
        ##bitmapGlyph = cast(glyph, ftglyph.FT_BitmapGlyph)
        ##gbmp = bitmapGlyph[0].bitmap
        return self.getBitmapArray()

    _ft_getGlyph = ftglyph.FT_Get_Glyph
    _ft_doneGlyph = staticmethod(ftglyph.FT_Done_Glyph)

    _glyph = None
    def getGlyph(self):
        glyph = self._glyph
        if glyph is None:
            glyph = ftglyph.FT_Glyph()
            self._ft_getGlyph(glyph)
            self._glyph = glyph
            def dealloc(wr, glyph=glyph):
                ftglyph.FT_Done_Glyph(glyph)
            weakref.ref(self, dealloc)
        return glyph
    glyph = property(getGlyph)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    bboxModes = dict(
        unscaled = 0,
        subpixels = 0,
        gridfit = 1,
        truncate = 2,
        pixels = 3,)

    _ft_getGlyphCBox = staticmethod(ftglyph.FT_Glyph_Get_CBox)
    def getCBox(self, mode=1):
        mode = self.bboxModes.get(mode, mode)

        cbox = FT.FT_BBox()
        self._ft_getGlyphCBox(self.glyph, mode, byref(cbox))
        return frombuffer(cbox, 'l').reshape((2,2))
    cbox = property(getCBox)

    @property
    def padvance(self):
        return self.advance >> 6
    def getPBox(self, mode=3):
        return self.getCBox(mode)
    pbox = property(getPBox)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def printInfo(self, out=None):
        print 'glyphIndex:', self.index, 'name:', repr(self.name)
        if self.numSubglyphs:
            print >> out, '    subglyphs:', self.numSubglyphs
        print >> out, '    advance:', (self.advance[0]>>6, self.advance[1]>>6), 'linear:', (self.linearHoriAdvance/ptDiv16, self.linearVertAdvance/ptDiv16)
        print '    (x, y), (w, h):', (self.bitmapLeft, self.bitmapTop), (self.bitmap.width, self.bitmap.rows)

        metrics = self.metrics
        print >> out, '    metrics:', (metrics.width/ptDiv, metrics.height/ptDiv)
        print >> out, '        hori:', (metrics.horiBearingX/ptDiv, metrics.horiBearingY/ptDiv, metrics.horiAdvance/ptDiv)
        print >> out, '        vert:', (metrics.vertBearingX/ptDiv, metrics.vertBearingY/ptDiv, metrics.vertAdvance/ptDiv)
        print >> out, '    cbox:', (self.getCBox(),)

Glyph = FreetypeGlyphSlot


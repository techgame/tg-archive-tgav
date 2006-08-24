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

from library import FreetypeLibrary
from glyph import FreetypeFaceGlyph

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeFace(object):
    faceIndex = 0
    loadFlags = FT.FT_LOAD_RENDER
    renderMode = FT.FT_Render_Mode.FT_RENDER_MODE_NORMAL

    #~ FreeType API interation ~~~~~~~~~~~~~~~~~~~~~~~~~~
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_Face

    _ft_new_face = staticmethod(FT.FT_New_Face)
    def __init__(self, fontFilename, faceIndex=0, ftLibrary=None):
        self._as_parameter_ = self._as_parameter_type_()
        ftLibrary = ftLibrary or FreetypeLibrary()
        self._ft_new_face(ftLibrary, fontFilename, faceIndex, byref(self._as_parameter_))
        self.library = ftLibrary

    _ft_done_face = FT.FT_Done_Face
    def __del__(self):
        if self._as_parameter_ is not None:
            self._ft_done_face()
        self._as_parameter_ = None

    @property
    def _face(self):
        return self._as_parameter_[0]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def numFaces(self):
        return self._face.num_faces

    @property
    def faceIndex(self):
        return self._face.face_index

    @property
    def faceFlags(self):
        return self._face.face_flags

    @property
    def styleFlags(self):
        return self._face.style_flags

    @property
    def numGlyphs(self):
        return self._face.num_glyphs

    @property
    def familyName(self):
        return self._face.family_name

    @property
    def styleName(self):
        return self._face.style_name

    @property
    def numFixedSizes(self):
        return self._face.num_fixed_sizes

    @property
    def availableSizes(self):
        return self._face.available_sizes

    @property
    def numCharmaps(self):
        return self._face.num_charmaps

    @property
    def charmap(self):
        return self._face.charmap

    @property
    def charmaps(self):
        return self._face.charmaps[:self.numCharmaps]

    @property
    def bbox(self):
        return self._face.bbox

    @property
    def unitsPerEM(self):
        return self._face.units_per_EM

    @property
    def ascender(self):
        return self._face.ascender

    @property
    def descender(self):
        return self._face.descender

    @property
    def height(self):
        return self._face.height

    @property
    def maxAdvanceWidth(self):
        return self._face.max_advance_width

    @property
    def maxAdvanceHeight(self):
        return self._face.max_advance_height

    @property
    def underlinePosition(self):
        return self._face.underline_position

    @property
    def underlineThickness(self):
        return self._face.underline_thickness

    @property
    def glyph(self):
        return FreetypeFaceGlyph(self._face.glyph, self)

    @property
    def size(self):
        return self._face.size

    @property
    def sizesList(self):
        result = []
        node = self._face.sizes_list.head
        while node:
            result.append(cast(node.data, FT.FT_Size))
            node = node.next
        return result

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_setCharSize = FT.FT_Set_Char_Size
    def setCharSize(self, size, dpi):
        if isinstance(size, tuple): 
            width, height = size
        else: width = height = size
        if isinstance(dpi, tuple): 
            wdpi, hdpi = dpi
        else: wdpi = hdpi = dpi
        self._ft_setCharSize(width, height, wdpi, hdpi)

    _ft_setPixelSizes = FT.FT_Set_Pixel_Sizes
    def setPixelSize(self, size):
        if isinstance(size, tuple): 
            width, height = size
        else: width = height = size
        self._ft_setPixelSizes(width, height)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_loadGlyph = FT.FT_Load_Glyph
    def loadGlyph(self, char, flags=None):
        if flags is None: 
            flags = self.loadFlags 
        self._ft_loadGlyph(ord(char), flags)
        return self.glyph

    def iterGlyphs(self, chars, flags=None):
        for char in chars:
            yield char, self.loadGlyph(char, flags)

    _ft_loadChar = FT.FT_Load_Char
    def loadChar(self, char, flags=None):
        if flags is None: 
            flags = self.loadFlags 
        self._ft_loadChar(ord(char), flags)
        return self.glyph

    def iterChars(self, chars, flags=None):
        for char in chars:
            yield char, self.loadChar(char, flags)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_setTransform = FT.FT_Set_Transform
    def setTransform(self, matrix, delta):
        self._ft_setTransform(matrix, delta)

    _ft_getKerning = FT.FT_Get_Kerning
    def getKerning(self, left, right, kernMode=None):
        aKerning = FT.FT_Vector()
        self._ft_getKerning(ord(left), ord(right), kernMode, byref(aKerning))
        return aKerning

    _ft_getGlyphName = FT.FT_Get_Glyph_Name
    def getGlyphName(self, char):
        buffer = ctypes.create_string_buffer(255)
        self._ft_getGlyphName(ord(char), buffer, len(buffer))
        return buffer.value

    _ft_getPostscriptName = FT.FT_Get_Postscript_Name
    def getPostscriptName(self):
        return self._ft_getPostscriptName()

    _ft_getCharIndex = FT.FT_Get_Char_Index
    def getCharIndex(self, char):
        return self._ft_getCharIndex(ord(char))

    _ft_getFirstChar = FT.FT_Get_First_Char
    def getFirstChar(self):
        glyphIndex = c_uint(0)
        charCode = self._ft_getFirstChar(byref(glyphIndex))
        return unichr(charCode), glyphIndex.value

    _ft_getNextChar = FT.FT_Get_Next_Char
    def getNextChar(self, char):
        glyphIndex = c_uint(0)
        charCode = self._ft_getNextChar(ord(char), byref(glyphIndex))
        return unichr(charCode), glyphIndex.value

    def iterCharCodes(self):
        glyphIndex = c_uint(0)
        glyphIndexRef = byref(glyphIndex)

        charCode = self._ft_getFirstChar(glyphIndexRef)
        while glyphIndex != 0:
            yield unichr(charCode), glyphIndex.value
            charCode = self._ft_getNextChar(charCode, glyphIndexRef)

    _ft_getNameIndex = FT.FT_Get_Name_Index
    def getNameIndex(self, name):
        return self._ft_getNameIndex(self, name)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    pass


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

import os
import weakref

import ctypes
from ctypes import byref, cast, c_void_p, c_uint

from raw import freetype as FT
from raw._ctypes_freetype import FreetypeException

from library import FreetypeLibrary
from glyph import FreetypeGlyphSlot

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ptDiv = float(1<<6)
ptDiv16 = float(1<<16)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeFace(object):
    faceIndex = 0
    GlyphType = FreetypeGlyphSlot

    #~ FreeType API interation ~~~~~~~~~~~~~~~~~~~~~~~~~~
    _as_parameter_ = None
    _as_parameter_type_ = FT.FT_Face

    _ft_new_face = staticmethod(FT.FT_New_Face)
    def __init__(self, fontFilename, faceIndex=0, ftLibrary=None):
        self.newFace(fontFilename, faceIndex, ftLibrary)

    def newFace(self, fontFilename, faceIndex=0, ftLibrary=None):
        ftLibrary = ftLibrary or FreetypeLibrary()
        self.filename = fontFilename

        _as_parameter_ = self._as_parameter_type_()
        self._ft_new_face(ftLibrary, fontFilename, faceIndex, byref(_as_parameter_))
        self._as_parameter_ = _as_parameter_

        def dealloc(wr, doneFace=FT.FT_Done_Face, pface=_as_parameter_):
            doneFace(pface)
        self._dealloc = weakref.ref(self, dealloc)

    _ft_done_face = FT.FT_Done_Face
    def close(self):
        if self._as_parameter_ is not None:
            del self._dealloc
            self._ft_done_face()
        self._as_parameter_ = None

    def __repr__(self):
        klass = self.__class__
        fmt = '<%s: %%s>' % (klass.__name__,)
        if not self._as_parameter_:
            info = '{uninitialized}'
        else:
            info = '"%(familyName)s(%(styleName)s):%(lastSize)s"' % self.getInfo()
        return fmt % (info,)

    def __contains__(self, key):
        if isinstance(key, basestring):
            return self.isCharAvailable(key)
        return False

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
    def flags(self):
        faceFlags = self.faceFlags
        return set(flag for mask, flag in self.faceFlagsMap.iteritems() if faceFlags & mask)

    def hasFlag(self, flag):
        flag = self.faceFlagsByName.get(flag, flag)
        return self.faceFlags & flag

    faceFlagsMap = {
        FT.FT_FACE_FLAG_SCALABLE: 'scalable',
        FT.FT_FACE_FLAG_FIXED_SIZES: 'fixed_sizes',
        FT.FT_FACE_FLAG_FIXED_WIDTH: 'fixed_wleadidth',
        FT.FT_FACE_FLAG_SFNT: 'sfnt',
        FT.FT_FACE_FLAG_HORIZONTAL: 'horizontal',
        FT.FT_FACE_FLAG_VERTICAL: 'vertical',
        FT.FT_FACE_FLAG_KERNING: 'kerning',
        FT.FT_FACE_FLAG_FAST_GLYPHS: 'fast_glyphs',
        FT.FT_FACE_FLAG_MULTIPLE_MASTERS: 'multiple_masters',
        FT.FT_FACE_FLAG_GLYPH_NAMES: 'glyph_names',
        FT.FT_FACE_FLAG_EXTERNAL_STREAM:'external_stream',
    }
    faceFlagsByName = dict((v,k) for k,v in faceFlagsMap.iteritems())

    @property
    def styleFlags(self):
        return self._face.style_flags

    @property
    def numGlyphs(self):
        return self._face.num_glyphs

    def getInfo(self):
        return dict(
                familyName=self.familyName, 
                postscriptName=self.postscriptName, 
                lastSize=self.lastSize,
                height=self.height / 64., 
                lineHeight=self.lineHeight / 64., 
                styleName=self.styleName)

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
        return self.GlyphType(self._face.glyph, self)

    @property
    def size(self):
        return self._face.size

    @property
    def lineHeight(self):
        return self._face.size[0].metrics.height

    @property
    def lineAscender(self):
        return self._face.size[0].metrics.ascender

    @property
    def lineDescender(self):
        return self._face.size[0].metrics.descender

    @property
    def sizesList(self):
        result = []
        node = self._face.sizes_list.head
        while node:
            result.append(cast(node.data, FT.FT_Size))
            node = node.next
        return result

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    lastSize = "{Not set}"
    def setSize(self, size, dpi=None):
        if dpi is None:
            return self.setPixelSize(size)
        else:
            return self.setCharSize(size, dpi)

    _ft_setCharSize = FT.FT_Set_Char_Size
    def setCharSize(self, size, dpi):
        if isinstance(size, tuple): 
            width, height = size
        else: width, height = 0, size
        if isinstance(dpi, tuple): 
            wdpi, hdpi = dpi
        else: wdpi = hdpi = dpi
        self._ft_setCharSize(int(width*ptDiv), int(height*ptDiv), wdpi, hdpi)
        self.lastSize = '%s@%sdpi'%(height, hdpi)

    _ft_setPixelSizes = FT.FT_Set_Pixel_Sizes
    def setPixelSize(self, size):
        if isinstance(size, tuple): 
            width, height = size
        else: width, height = 0, size
        self._ft_setPixelSizes(width, height)
        self._lastSize = (width or height, height, None, None)
        self.lastSize = str(height)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    glyphLoadFlags = FT.FT_LOAD_RENDER
    glyphRenderMode = FT.FT_Render_Mode.FT_RENDER_MODE_NORMAL

    def allowVerticalLayout(self, onlyIfVertical=True):
        if 'vertical' in self.flags:
            self.loadVerticalLayout()
            return True
        return False

    def loadVerticalLayout(self):
        self.glyphLoadFlags |= FT.FT_LOAD_VERTICAL_LAYOUT

    _ft_loadGlyph = FT.FT_Load_Glyph
    def loadGlyph(self, glyphIndex, flags=None):
        if flags is None: 
            flags = self.glyphLoadFlags 
        if isinstance(glyphIndex, basestring):
            glyphIndex = self.getCharIndex(glyphIndex)
        self._ft_loadGlyph(glyphIndex, flags)
        glyph = self.glyph
        glyph.index = glyphIndex
        return glyph
    __getitem__ = loadGlyph

    def iterUniqueGlyphs(self, chars, flags=None):
        indexes = frozenset(self.iterCharIndexes(chars))
        for glyphIndex in indexes:
            glyph = self.loadGlyph(glyphIndex, flags)
            yield glyphIndex, glyph

    def iterGlyphs(self, chars, flags=None):
        for char, glyphIndex in self.iterCharIndexes(chars, True):
            glyph = self.loadGlyph(glyphIndex, flags)
            if glyph or char in (0, 'x00'):
                yield char, glyph

    _ft_loadChar = FT.FT_Load_Char
    def loadChar(self, char, flags=None):
        if flags is None: 
            flags = self.glyphLoadFlags 
        self._ft_loadChar(ord(char), flags)
        glyph = self.glyph
        glyph.index = self.getCharIndex(char)
        return glyph

    def iterChars(self, chars, flags=None):
        for char in chars:
            glyph = self.loadChar(char, flags)
            if glyph or char == '\x00':
                yield char, glyph

    _ft_getKerning = FT.FT_Get_Kerning
    def getKerning(self, left, right, kernMode=0):
        aKerning = FT.FT_Vector()
        if isinstance(left, basestring):
            left = self.getCharIndex(left)
        if isinstance(right, basestring):
            right = self.getCharIndex(right)
        self._ft_getKerning(left, right, kernMode, byref(aKerning))
        return (aKerning.x, aKerning.y)
    def getKerningByIndex(self, leftIndex, rightIndex, kernMode=0):
        aKerning = FT.FT_Vector()
        self._ft_getKerning(leftIndex, rightIndex, kernMode, byref(aKerning))
        return (aKerning.x, aKerning.y)

    def iterKerning(self, chars, kernMode=0):
        left = None
        for right in self.iterCharIndexes(chars):
            if left is None:
                yield (0, 0)
            else:
                self._ft_getKerning(left, right, kernMode, byref(aKerning))
                yield (aKerning.x, aKerning.y)
            left = right

    def iterKerningSwapped(self, chars, kernMode=0):
        right = None
        for left in self.iterCharIndexes(chars):
            if right is None:
                yield (0, 0)
            else:
                self._ft_getKerning(left, right, kernMode, byref(aKerning))
                yield (aKerning.x, aKerning.y)
            right = left

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_setTransform = FT.FT_Set_Transform
    def setTransform(self, matrix, delta):
        self._ft_setTransform(matrix, delta)

    _ft_getPostscriptName = FT.FT_Get_Postscript_Name
    def getPostscriptName(self):
        return self._ft_getPostscriptName()
    postscriptName = property(getPostscriptName)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_selectCharmap = FT.FT_Select_Charmap
    def selectCharmap(self, encoding):
        self._ft_selectCharmap(encoding)

    _ft_setCharmap = FT.FT_Set_Charmap
    def getCharmap(self):
        return self.charmap
    def setCharmap(self, charmap):
        self._ft_setCharmap(charmap)

    _ft_getCharmapIndex = staticmethod(FT.FT_Get_Charmap_Index)
    def getCharmapIndex(self, charmap):
        return self._ft_getCharmapIndex(charmap)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _ft_getCharIndex = FT.FT_Get_Char_Index
    def getCharIndex(self, char):
        return self._ft_getCharIndex(ord(char))
    getOrdinalIndex = _ft_getCharIndex

    def uniqueCharIndexSet(self, chars):
        return frozenset(self.iterCharIndexes(chars, False))
    def iterCharIndexes(self, chars=None, bMapping=False):
        if not chars:
            return self.iterAllChars(bMapping)
        elif bMapping:
            return ((char, self._ft_getCharIndex(ord(char))) for char in chars)
        else:
            return (self._ft_getCharIndex(ord(char)) for char in chars)

    def charIndexMap(self, chars, mapping=None):
        if mapping is None: 
            mapping = dict()
        mapping.update(self.iterCharIndexes(chars, True))
        return mapping

    def isCharAvailable(self, char):
        return 0 < self.getCharIndex(char)

    _ft_getGlyphName = FT.FT_Get_Glyph_Name
    def getGlyphName(self, glyphIndex):
        if isinstance(glyphIndex, basestring):
            glyphIndex = self.getCharIndex(glyphIndex)
        buffer = ctypes.create_string_buffer(255)
        self._ft_getGlyphName(glyphIndex, buffer, len(buffer))
        return buffer.value

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

    def iterAllChars(self, bMapping=False):
        glyphIndex = c_uint(0)
        glyphIndexRef = byref(glyphIndex)

        charCode = self._ft_getFirstChar(glyphIndexRef)
        if bMapping:
            while glyphIndex:
                yield unichr(charCode), glyphIndex.value
                charCode = self._ft_getNextChar(charCode, glyphIndexRef)
        else:
            while glyphIndex:
                yield unichr(charCode)
                charCode = self._ft_getNextChar(charCode, glyphIndexRef)

    _ft_getNameIndex = FT.FT_Get_Name_Index
    def getNameIndex(self, name):
        return self._ft_getNameIndex(name)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def printInfo(self, out=None):
        face = self
        print >> out, 'name:', face.getPostscriptName(), 'family:', face.familyName, 'style:', face.styleName
        print >> out, '  faces:', face.numFaces, '(%s)' % (face.faceIndex,), 'glyph count:', face.numGlyphs
        print >> out, '  flags:', hex(face.faceFlags), '=', ' | '.join(face.flags)
        print >> out, '  metrics:'
        print >> out, '    units per em:', face.unitsPerEM/ptDiv
        print >> out, '    ascender:', face.ascender / ptDiv, 'descender:', face.descender / ptDiv, 'height:', face.height / ptDiv, '(a-d:', (face.ascender-face.descender)/ptDiv, ')'
        print >> out, '    lineAscender:', face.lineAscender / ptDiv, 'lineDescender:', face.lineDescender / ptDiv, 'lineHeight:', face.lineHeight / ptDiv, '(la-ld:', (face.lineAscender-face.lineDescender)/ptDiv, ')'
        print >> out, '    bbox:', [(face.bbox.xMin/ptDiv, face.bbox.yMin/ptDiv), (face.bbox.xMax/ptDiv, face.bbox.yMax/ptDiv)]
        print >> out, '    underline pos:', face.underlinePosition/ptDiv, 'thickness:', face.underlineThickness/ptDiv
        print >> out, '    max advance width:', face.maxAdvanceWidth/ptDiv, 'height:', face.maxAdvanceHeight/ptDiv
        cm = face.charmap[0]
        print >> out, '  charmaps:'
        print >> out, '    current id:', cm.encoding.value, 'index:', face.getCharmapIndex(face.charmap), 'plat_id:', cm.platform_id, 'encoding_id:',cm.encoding_id
        print >> out, '    others(%s):' % (face.numCharmaps,)
        for index, cm in enumerate(face.charmaps[:face.numCharmaps]):
            cm = cm[0]
            print >> out, '      id:', cm.encoding.value, 'index:', index, 'plat_id:', cm.platform_id, 'encoding_id:', cm.encoding_id
        print >> out

Face = FreetypeFace

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeFaceIndex(dict):
    FaceFactory = FreetypeFace
    primaryKeys = ['filename', 'familyName']

    systemFontPaths = {
        'Windows': [r'%SystemRoot%\Fonts'],
        'Darwin': [r'/System/Library/Fonts', r'/Library/Fonts', r'~/Library/Fonts']
        }
    _normStyleName = staticmethod(str.lower)

    def __init__(self, path=None, sysPaths=False, primaryKeys=None):
        if primaryKeys is not None:
            self.primaryKeys = primaryKeys
        dict.__init__(self)
        if path is not None:
            self.addPath(path)

        if sysPaths:
            import platform
            systemPath = self.systemFontPaths[platform.system()]
            self.addPath(systemPath)

    def face(self, key, style=0):
        if isinstance(key, dict):
            faceEntry = key

        faceEntry = self[key]

        if isinstance(style, basestring):
            style = self._normStyleName(style)
            style = faceEntry['styles'].index(style)

        face = self.FaceFactory(faceEntry['filename'], style)
        return face

    @classmethod
    def forPath(klass, path):
        return klass(path)

    @classmethod
    def forSystem(klass):
        return klass(sysPaths=True)

    def addPath(self, path):
        if isinstance(path, basestring):
            paths = [path]
        else: paths = iter(path)

        for each in paths:
            each = os.path.expandvars(each)
            each = os.path.expanduser(each)

            for fname in os.listdir(each):
                fontFilename = os.path.join(each, fname)
                if os.path.isfile(fontFilename):
                    self.add(fontFilename)

    def add(self, fontFilename):
        try:
            face0 = self.FaceFactory(fontFilename)
        except FreetypeException:
            return False

        self.addFace(face0, fontFilename)
        return True

    def addFace(self, face0, fontFilename):
        faceSet = [face0] + [self.FaceFactory(fontFilename,idx) for idx in range(1, face0.numFaces)]
        self.addFaceSet(faceSet)

    def addFaceSet(self, faceSet):
        face0 = faceSet[0]

        entry = {
            'filename': face0.filename,
            'family': face0.familyName,
            'postscript': face0.postscriptName,
            'styles': map(self._normStyleName, [facei.styleName for facei in faceSet]),
            'flags': face0.flags,
            }

        for pkey in self.primaryKeys:
            pv = getattr(face0, pkey)
            self[pv] = entry

FaceIndex = FreetypeFaceIndex

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    fi = FaceIndex(None, True, ['familyName'])
    for n, e in fi.iteritems():
        print '%30r: [%s]' % (n, ', '.join(e['styles']))
        print '%30s  [%s]' % ('', ', '.join(e['flags']))

    try: 
        f = fi.face('Courier')
        print
        f.printInfo()
        print
    except LookupError: 
        pass

    try: 
        f = fi.face('Comic Sans MS')
        print
        f.printInfo()
        print
    except LookupError: 
        pass



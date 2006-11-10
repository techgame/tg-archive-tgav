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

from ctypes import c_void_p
from numpy import ndarray, array, uint32

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

drawModeMap = {
    None: gl.GL_POINTS,
    gl.GL_POINTS: gl.GL_POINTS,
    'points': gl.GL_POINTS,

    gl.GL_LINES: gl.GL_LINES,
    'lines': gl.GL_LINES,
    gl.GL_LINE_LOOP: gl.GL_LINE_LOOP,
    'lineLoop': gl.GL_LINE_LOOP,
    gl.GL_LINE_STRIP: gl.GL_LINE_STRIP,
    'lineStrip': gl.GL_LINE_STRIP,

    gl.GL_TRIANGLES: gl.GL_TRIANGLES,
    'triangles': gl.GL_TRIANGLES,
    gl.GL_TRIANGLE_STRIP: gl.GL_TRIANGLE_STRIP,
    'triangleStrip': gl.GL_TRIANGLE_STRIP,
    gl.GL_TRIANGLE_FAN: gl.GL_TRIANGLE_FAN,
    'triangleFan': gl.GL_TRIANGLE_FAN,

    gl.GL_QUADS: gl.GL_QUADS,
    'quads': gl.GL_QUADS,
    gl.GL_QUAD_STRIP: gl.GL_QUAD_STRIP,
    'quadStrip': gl.GL_QUAD_STRIP,
    }

dataFormatMap = {
    gl.GL_BYTE: gl.GL_BYTE, 
    'char': gl.GL_BYTE,
    'byte': gl.GL_BYTE,
    'int8': gl.GL_BYTE,

    gl.GL_SHORT: gl.GL_SHORT,
    'short': gl.GL_SHORT,
    'int16': gl.GL_SHORT,

    gl.GL_INT: gl.GL_INT,
    'int': gl.GL_INT,
    'int32': gl.GL_INT,

    gl.GL_UNSIGNED_BYTE: gl.GL_UNSIGNED_BYTE,
    'uint8': gl.GL_UNSIGNED_BYTE,
    'ubyte': gl.GL_UNSIGNED_BYTE,
    'uchar': gl.GL_UNSIGNED_BYTE,
    'unsignedByte': gl.GL_UNSIGNED_BYTE,

    gl.GL_UNSIGNED_SHORT: gl.GL_UNSIGNED_SHORT,
    'ushort': gl.GL_UNSIGNED_SHORT,
    'unsignedShort': gl.GL_UNSIGNED_SHORT,
    'uint16': gl.GL_UNSIGNED_SHORT,

    gl.GL_UNSIGNED_INT: gl.GL_UNSIGNED_INT,
    'uint': gl.GL_UNSIGNED_INT,
    'unsignedInt': gl.GL_UNSIGNED_INT,
    'uint32': gl.GL_UNSIGNED_INT,
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawBlockBase(object):
    mode = None
    modeMap = drawModeMap
    dataFormatMap = dataFormatMap

    def setMode(self, mode=None):
        self.mode = self.modeMap[mode or self.mode]

    @staticmethod
    def asElementArray(data, dtype=None):
        return array(data, dtype, copy=False, order='C', ndmin=1, subok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawArrays(DrawBlockBase):
    first = 0
    count = 0

    def __init__(self, mode=None, first=0, count=None):
        self.setMode(mode)
        self.setSpan(first, count)

    def setSpan(self, first=0, count=None):
        if first > 0 and count is None:
            count = first
            first = 0

        self.first = first
        self.count = count

    glDrawArrays = staticmethod(gl.glDrawArrays)
    def render(self):
        self.glDrawArrays(self.mode, self.first, self.count)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawArrays(DrawBlockBase):
    firstArray = None
    countArray = None

    def __init__(self, mode=None, firstList=None, countList=None):
        self.setMode(mode)
        if firstList or countList:
            self.setArrays(firstList, countList)

    def setArrays(self, firstList, countList):
        if len(firstList) != len(countList):
            raise ArgumentError("firstList (%s) and countList (%s) do not contain the same number of elements." % (len(firstList), len(countList)))

        self.firstArray = self.asElementArray(firstList, uint32)
        self.countArray = self.asElementArray(countList, uint32)

    glMultiDrawArrays = staticmethod(gl.glMultiDrawArrays)
    def render(self):
        countArray = self.countArray
        self.glMultiDrawArrays(self.mode, self.firstArray.ctypes, countArray.ctypes, len(countArray))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawElements(ndarray, DrawBlockBase):
    dataFormat = None
    indicies = None

    def __new__(klass, mode=None, data=None, dtype=None, copy=False):
        data = array(data, dtype, copy=copy, order='C', ndmin=1)
        return data.view(klass)

    def __init__(self, mode=None, data=None, dtype=None, copy=False):
        self._config()
        self.setMode(mode)

    def _config(self):
        self._as_parameter_ = self.ctypes._as_parameter_
        self._inferDataFormat()

    def _inferDataFormat(self):
        self.dataFormat = self.dataFormatMap[self.dtype.name]

    glDrawElements = staticmethod(gl.glDrawElements)
    def render(self):
        self.glDrawElements(self.mode, len(self), self.dataFormat, self)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawRangeElements(DrawElements):
    start = 0
    end = 0

    def __new__(klass, mode=None, start=0, end=0, data=None, dtype=None, copy=False):
        data = array(data, dtype, copy=copy, order='C', ndmin=1)
        return data.view(klass)

    def __init__(self, mode=None, start=0, end=0, data=None, dtype=None, copy=False):
        DrawElements.__init__(self, mode, data, dtype, copy)
        self.setRange(start, end)

    def setRange(self, start, end):
        self.start = start
        self.end = end

    glDrawRangeElements = staticmethod(gl.glDrawRangeElements)
    def render(self):
        self.glDrawRangeElements(self.mode, self.start, self.end, len(self), self.dataFormat, self)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawElementsBase(DrawBlockBase):
    def __init__(self, mode=None, entries=None):
        self.createEntries()
        self.setMode(mode)
        if entries is not None:
            self.updateEntries(entries)

    _cacheInvalid = True
    _entries = None
    def getEntries(self, invalidateCache=True):
        if invalidateCache:
            self._cacheInvalid = True
        return self._entries
    def setEntries(self, entries):
        self._entries = entries
        self._cacheInvalid = True
    entries = property(getEntries, setEntries)
        
    def updateEntries(self, entries):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def createEntries(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def _safeGetIndicesList(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    _indicesCountCache = None
    _indicesListCache = None
    def _cacheIndicesList(self):
        indicesList = self._safeGetIndicesList()

        CountArrayType = (gl.GLsizei*len(indicesList))
        self._indicesCountCache = CountArrayType(*[len(indices) for indices in indicesList])

        IndicesListType = (c_void_p*len(indicesList))
        self._indicesListCache = IndicesListType(*[indices.ctypes for indices in indicesList])

        self._cacheInvalid = False

    glMultiDrawElements = staticmethod(gl.glMultiDrawElements)
    def render(self):
        if self._cacheInvalid:
            self._cacheIndicesList()

        indicesListCache = self._indicesListCache
        self.glMultiDrawElements(self.mode, self._indicesCountCache, self.dataFormat, indicesListCache, len(indicesListCache))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawElementList(MultiDrawElementsBase):
    def createEntries(self):
        self._entries = list()
    def _safeGetIndicesList(self):
        return self._entries
    def updateEntries(self, entries):
        self.entries.update(entries)

class MultiDrawElementSet(MultiDrawElementsBase):
    def createEntries(self):
        self._entries = set()
    def _safeGetIndicesList(self):
        return list(self._entries)
    def updateEntries(self, entries):
        self.entries.update(entries)

class MultiDrawElementDict(MultiDrawElementsBase):
    def createEntries(self):
        self._entries = dict()
    def _safeGetIndicesList(self):
        return self._entries.values()
    def updateEntries(self, entries):
        self.entries.extend(entries)


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
from numpy import asarray, uint32

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawBlockBase(object):
    mode = None
    modeMap = {
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

    dataFormatMap = dict(
        uint8=gl.GL_UNSIGNED_BYTE,
        uint16=gl.GL_UNSIGNED_SHORT,
        uint32=gl.GL_UNSIGNED_INT,
        )

    def setMode(self, mode=None):
        self.mode = self.modeMap[mode or self.mode]

    @staticmethod
    def asElementArray(data, dtype=None):
        return asarray(data, dtype, copy=False, order='C', ndmin=1, subok=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawArrays(DrawBlockBase):
    first = 0
    count = 0

    def __init__(self, mode=None, first=0, count=0):
        self.setMode(mode)
        self.setRange(first, count)

    def setRange(self, first, count):
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

class DrawElements(DrawBlockBase):
    dataFormat = None
    indicies = None

    def __init__(self, mode=None, indices=None, dataFormat=None):
        self.setMode(mode)
        if indices:
            self.setIndices(indices, dataFormat)
        elif dataFormat:
            self.dataFormat = dataFormat

    _indices = None
    def getIndices(self):
        return self._indices
    def setIndices(self, indices, dataFormat=None):
        dataFormat = dataFormat or self.dataFormat
        self._indices = self.asElementArray(indices, dataFormat)
        if not dataFormat:
            dataFormat = dataFormat or self._inferDataFormat()
        self.dataFormat = dataFormat
    indices = property(getIndices, setIndices)

    def _inferDataFormat(self):
        return self.dataFormatMap[self._indices.dtype.name]

    glDrawElements = staticmethod(gl.glDrawElements)
    def render(self):
        indices = self._indices
        self.glDrawElements(self.mode, len(indices), self.dataFormat, indices.ctypes)

class DrawRangeElements(DrawElements):
    first = 0
    count = 0

    def __init__(self, mode=None, start=0, count=0, indices=None, dataFormat=None):
        DrawElements.__init__(self, mode, indices, dataFormat)
        self.setRange(start, count)

    def setRange(self, first, count):
        self.first = first
        self.count = count

    glDrawRangeElements = staticmethod(gl.glDrawRangeElements)
    def render(self):
        self.glDrawRangeElements(self.mode, self.start, self.count, len(indices), self.dataFormat, indices.ctypes)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawElements(DrawBlockBase):
    def __init__(self, mode=None, indices=None):
        self.indices = indices

    _cacheInvalid = True
    _indices = None
    def getIndices(self, invalidateCache=True):
        if invalidateCache:
            self._cacheInvalid = True
        return self._indices
    def setIndices(self, indices):
        self._indices = indices
        self._cacheInvalid = True
    indices = property(getIndices, setIndices)
        
    def _safeGetIndicesList(self):
        return self._iterIndices

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


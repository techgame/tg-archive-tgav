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

from TG.openGL.drawArrays import DrawBlockBase

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawBufferElements(DrawBlockBase):
    count = 0

    def __init__(self, mode=None, count=0, bufferOffset=None):
        self.setMode(mode)
        self.count = count
        self.setOffset(bufferOffset)

    _p_offset = c_void_p(0)
    def getOffset(self):
        return self._p_offset.value
    def setOffset(self, bufferOffset):
        self._p_offset = c_void_p(bufferOffset)
    offset = property(getOffset, setOffset)

    glDrawElements = staticmethod(gl.glDrawElements)
    def render(self):
        self.glDrawElements(self.mode, self.count, self.dataFormat, self._p_offset)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DrawBufferRangeElements(DrawBufferElements):
    start = 0
    end = 0

    def __init__(self, mode=None, start=0, end=0, count=0, bufferOffset=None):
        DrawBufferElements.__init__(self, mode, count, bufferOffset)
        self.setRange(start, end)

    def setRange(self, start, end):
        self.start = start
        self.end = end

    glDrawRangeElements = staticmethod(gl.glDrawRangeElements)
    def render(self):
        self.glDrawRangeElements(self.mode, self.start, self.end, self.count, self.dataFormat, self._p_offset)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawBufferElementsBase(DrawBlockBase):
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
        
    def createEntries(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def _safeGetCountOffsetList(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    _countsListCache = None
    _offsetsListCache = None
    def _cacheIndicesList(self):
        countOffsetList = self._safeGetCountOffsetList()

        CountArrayType = (gl.GLsizei*len(indicesList))
        self._countsListCache = CountArrayType(*[count for count, offset in indicesList])

        IndicesListType = (c_void_p*len(indicesList))
        self._offsetsListCache = IndicesListType(*[offset for count, offset in indicesList])

        self._cacheInvalid = False

    glMultiDrawElements = staticmethod(gl.glMultiDrawElements)
    def render(self):
        if self._cacheInvalid:
            self._cacheIndicesList()

        offsetsListCache = self._offsetsListCache
        self.glMultiDrawElements(self.mode, self._countsListCache, self.dataFormat, offsetsListCache, len(offsetsListCache))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiDrawBufferElementList(MultiDrawBufferElementsBase):
    def createEntries(self):
        self._entries = list()
    def _safeGetCountOffsetList(self):
        return self._entries
    def updateEntries(self, entries):
        self.entries.update(entries)

class MultiDrawBufferElementSet(MultiDrawBufferElementsBase):
    def createEntries(self):
        self._entries = set()
    def _safeGetCountOffsetList(self):
        return list(self._entries)
    def updateEntries(self, entries):
        self.entries.update(entries)

class MultiDrawBufferElementDict(MultiDrawBufferElementsBase):
    def createEntries(self):
        self._entries = dict()
    def _safeGetCountOffsetList(self):
        return self._entries.values()
    def updateEntries(self, entries):
        self.entries.extend(entries)


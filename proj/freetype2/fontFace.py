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

from itertools import izip
from face import FreetypeFace

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variiables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ptDiv = float(1<<6)
ptDiv16 = float(1<<16)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _asFreq(d):
    freq = dict()
    for e in d: 
        freq[e] = freq.get(e, 0) + 1
    return ', '.join('%s:%s' % e for e in freq.iteritems())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FreetypeFontFace(object):
    FreetypeFaceFactory = FreetypeFace

    def __init__(self, filename, fontSize=None):
        self.setFilename(filename)
        if fontSize:
            self.setFontSize(fontSize)

    _filename = ''
    def getFilename(self):
        return self._filename
    def setFilename(self, filename):
        self._filename = filename
        self.loadFaceFromFilename(filename)
    filename = property(getFilename, setFilename)

    def loadFaceFromFilename(self, filename):
        face = self.FreetypeFaceFactory(filename)
        self.setFace(face)

    face = None
    def getFace(self):
        return self.face
    def setFace(self, face):
        if self.face is not None:
            self._delFace(self.face)
        self.face = face
        self._initFace(self.face)

    def _initFace(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def _delFace(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    fontSize = 16
    def getFontSize(self):
        return self.fontSize
    def setFontSize(self, fontSize):
        self.fontSize = fontSize
        self._setFaceSize(fontSize)
    def _setFaceSize(self, fontSize):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))


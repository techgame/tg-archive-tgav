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

from .raw import gl, glu

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Selector(object):
    def __init__(self):
        self.selection = []

    def __iter__(self):
        return iter(self.selection)

    def __enter__(self):
        self._namedItems = {}

    def __exit__(self, exc=None, excType=None, excTb=None):
        del self._namedItems

    def setItem(self, item):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def pushItem(self, item):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def popItem(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def renderPickMatrix(self, vpbox):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NameSelector(Selector):
    """Uses the builtin name-based geometry selection model provided by OpenGL"""

    def __init__(self, pos, size=(2,2), bufferSize=1024):
        Selector.__init__(self)
        self.pos = pos
        self.size = size
        self.setBufferSize(bufferSize)

    def __enter__(self):
        Selector.__enter__(self)
        gl.glSelectBuffer(self.bufferSize, self._buffer)
        gl.glRenderMode(gl.GL_SELECT)
        gl.glInitNames()
        gl.glPushName(0)

    def __exit__(self, exc=None, excType=None, excTb=None):
        hitRecords = gl.glRenderMode(gl.GL_RENDER)
        self.selection = self._processHits(hitRecords, self._namedItems)
        return Selector.__exit__(self, exc, excType, excTb)

    def _processHits(self, hitRecords, namedItems):
        offset = 0
        buffer = self._buffer
        result = []
        for hit in xrange(hitRecords):
            nameRecords, minZ, maxZ = buffer[offset:offset+3]
            names = list(buffer[offset+3:offset+3+nameRecords])
            offset += 3+nameRecords

            namedHit = ((minZ, maxZ), [namedItems[n] for n in names])
            result.append(namedHit)
        return result

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getBufferSize(self):
        return len(self._buffer)
    def setBufferSize(self, size):
        self._buffer = (gl.GLuint*size)()
    bufferSize = property(getBufferSize, setBufferSize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def setItem(self, item):
        n = id(item)
        self._namedItems[n] = item
        gl.glLoadName(n)
    def pushItem(self, item):
        n = id(item)
        self._namedItems[n] = item
        gl.glPushName(n)
    def popItem(self):
        gl.glPopName()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderPickMatrix(self, vpbox):
        p = self.pos
        s = self.size
        vp = vpbox.pos[:2].tolist()
        vs = vpbox.size[:2].tolist()
        viewport = (gl.GLint*4)(*(vp+vs))

        glu.gluPickMatrix(p[0], p[1], s[0], s[1], viewport)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorSelector(Selector):
    """Runs a selection pass using the color buffer to allow for fragments to
    affect what gets picked.  This enables textures (and alpha) to affect the
    shape of objects."""


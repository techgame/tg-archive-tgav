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
    def start(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def finish(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def setItem(self, item):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def pushItem(self, item):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def popItem(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class NameSelector(Selector):
    """Uses the builtin name-based geometry selection model provided by OpenGL"""

    def __init__(self, bufferSize=1024):
        Selector.__init__(self)
        self.setBufferSize(bufferSize)
        self._namedItems = {}

    _buffer = (gl.GLuint*0)()
    def getBufferSize(self):
        return len(self._buffer)
    def setBufferSize(self, size):
        self._buffer = (gl.GLuint*size)()
    bufferSize = property(getBufferSize, setBufferSize)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def start(self):
        self._namedItems.clear()
        gl.glSelectBuffer(self.bufferSize, self._buffer)
        gl.glRenderMode(gl.GL_SELECT)
        gl.glInitNames()

    def finish(self):
        hitRecords = gl.glRenderMode(gl.GL_RENDER)
        selection = self._processHits(hitRecords, self._namedItems)
        self._namedItems.clear()
        return selection

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

    c_viewport = (gl.GLint*4)
    def pickMatrix(self, pos, size, vpbox):
        viewport = self.c_viewport(*vpbox)
        glu.gluPickMatrix(pos[0], pos[1], size[0], size[1], viewport)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorSelector(Selector):
    """Runs a selection pass using the color buffer to allow for fragments to
    affect what gets picked.  This enables textures (and alpha) to affect the
    shape of objects."""


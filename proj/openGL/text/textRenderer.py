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

from TG.openGL.data.bufferObjects import ArrayBuffer

from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextDisplay(object):
    mode = gl.GL_QUADS
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    def update(self, textObj, textData, geometry):
        self.geometry = geometry
        self.texture = textData.texture

    def render(self):
        if 0:
            self.texture.deselect()
            glColor4f(1., 1., 1., .1)
            self.geometry.draw()
            glColor4f(1., 1., 1., 1.)
        self.texture.select()
        self.geometry.draw()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextBufferedDisplay(object):
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    buffer = None
    def update(self, textObj, textData, geometry):
        self.geometry = geometry
        self.texture = textData.texture

        buff = self.buffer 
        if buff is None:
            buff = ArrayBuffer('dynamicDraw')
            self.buffer = buff

        buff.bind()
        buff.sendData(geometry)
        buff.unbind()

    glInterleavedArrays = staticmethod(gl.glInterleavedArrays)
    def render(self, glColor4f=gl.glColor4f):
        buf = self.buffer
        buf.bind()
        if 0:
            self.texture.deselect()
            glColor4f(1., 1., 1., .1)
            self.geometry.draw(vboOffset=0)
            glColor4f(1., 1., 1., 1.)

        self.texture.select()
        self.geometry.draw(vboOffset=0)
        buf.unbind()
    __call__ = render


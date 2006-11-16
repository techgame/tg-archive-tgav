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
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    def update(self, textObj, textData, geometry):
        self.geometry = geometry
        self.texture = textData.texture

    glDepthMask = staticmethod(gl.glDepthMask)
    def render(self):
        glDepthMask = self.glDepthMask
        glDepthMask(False)
        self.texture.select()
        self.geometry.draw()
        glDepthMask = self.glDepthMask
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

        # push the data to the card
        buff = self.buffer 
        if buff is None:
            buff = ArrayBuffer('dynamicDraw')
            self.buffer = buff

        buff.bind()
        buff.sendData(geometry)
        buff.unbind()

    glDepthMask = staticmethod(gl.glDepthMask)
    def render(self, glColor4f=gl.glColor4f):
        glDepthMask = self.glDepthMask
        glDepthMask(False)
        buf = self.buffer
        buf.bind()
        self.texture.select()
        self.geometry.draw(vboOffset=0)
        buf.unbind()
        glDepthMask(True)
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Display Mode Map
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

displayFactoryMap = {
    'unbuffered': TextDisplay,
    'buffered': TextBufferedDisplay,
    }
displayFactoryMap[None] = displayFactoryMap['buffered']


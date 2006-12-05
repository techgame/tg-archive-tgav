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
        self.color = textObj.color
        self.texture = textData.texture

    def render(self):
        self.texture.select()
        self.color.select()

        geom = self.geometry
        gl.glInterleavedArrays(geom.gltypeid, 0, geom.ctypes)
        gl.glDrawArrays(geom.drawMode, 0, geom.size)
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextBufferedDisplay(object):
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    buffer = None
    def update(self, textObj, textData, geometry):
        self.geometry = geometry
        self.color = textObj.color
        self.texture = textData.texture

        # push the data to the card
        buff = self.buffer 
        if buff is None:
            buff = ArrayBuffer('dynamicDraw')
            self.buffer = buff

        buff.bind()
        buff.sendData(geometry)
        buff.unbind()

    def render(self, glColor4f=gl.glColor4f):
        buf = self.buffer
        buf.bind()
        self.texture.select()
        self.color.select()

        geom = self.geometry
        gl.glInterleavedArrays(geom.gltypeid, 0, 0)
        gl.glDrawArrays(geom.drawMode, 0, geom.size)

        buf.unbind()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Display Mode Map
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

displayFactoryMap = {
    'unbuffered': TextDisplay,
    'buffered': TextBufferedDisplay,
    }
displayFactoryMap[None] = displayFactoryMap['buffered']


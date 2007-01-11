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

from functools import partial

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

        self.glInterleavedArrays = partial(gl.glInterleavedArrays, geometry.glTypeId, 0, geometry.ctypes)
        self.glDrawArrays = partial(gl.glDrawArrays, geometry.drawMode, 0, geometry.size)

    def render(self):
        self.texture.select()

        self.glInterleavedArrays()
        self.glDrawArrays()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextBufferedDisplay(object):
    def __init__(self, *args, **kw):
        if args or kw:
            self.update(*args, **kw)

    buffer = None
    def update(self, textObj, textData, geometry):
        self.geometry = geometry
        #self.color = textObj.color
        self.texture = textData.texture

        # push the data to the card
        buff = self.buffer 
        if buff is None:
            buff = ArrayBuffer('dynamicDraw')
            self.buffer = buff

        buff.bind()
        buff.sendData(geometry)
        buff.unbind()

        self.glInterleavedArrays = partial(gl.glInterleavedArrays, geometry.glTypeId, 0, 0)
        self.glDrawArrays = partial(gl.glDrawArrays, geometry.drawMode, 0, geometry.size)

    def render(self, glColor4f=gl.glColor4f):
        #self.color.select()
        self.texture.select()

        buff = self.buffer
        buff.bind()

        self.glInterleavedArrays()
        self.glDrawArrays()

        ## Font rect/outline highlighting code:
        #if 0:
        #    self.texture.deselect()
        #    gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        #    gl.glColor4f(1, .9, .9, .2)
        #    gl.glDrawArrays(geom.drawMode, 0, geom.size)

        #    gl.glColor4f(1, .4, .4, .4)
        #    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        #    gl.glDrawArrays(geom.drawMode, 0, geom.size)
        #    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        #    gl.glColor4f(1, 1, 1, 1)

        buff.unbind()
    __call__ = render

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Display Mode Map
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

displayFactoryMap = {
    'unbuffered': TextDisplay,
    'buffered': TextBufferedDisplay,
    }
displayFactoryMap[None] = displayFactoryMap['buffered']


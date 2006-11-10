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

from numpy import array

from TG.openGL.text import textWrapping
from TG.openGL.data.bufferObjects import ArrayBuffer
from TG.openGL.raw import gl

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Layout
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextLayout(object):
    align = 0
    wrapper = textWrapping.BasicTextWrapper()
    
    def layout(self, textData, **kw):
        geo = self.layoutText(textData, **kw)
        def lfn(tex=textData.texture, geo=geo):
            #gl.glColor4f(1., 1., 1., .1)
            #geo.draw(gl.GL_QUADS)
            #gl.glColor4f(1., 1., 1., 1.)
            tex.select()
            geo.draw(gl.GL_QUADS)
            tex.deselect()
        return geo, lfn

    def layoutBuffer(self, textData, **kw):
        geo = self.layoutText(textData, **kw)

        ab = ArrayBuffer()
        ab.sendData(geo)
        def lfn(tex=textData.texture, ab=ab, dataFormat=geo.dataFormat, count=len(geo.flat)):
            ab.bind()
            gl.glInterleavedArrays(dataFormat, 0, 0)

            #gl.glColor4f(1., 1., 1., .1)
            #gl.glDrawArrays(gl.GL_QUADS, 0, count)
            #gl.glColor4f(1., 1., 1., 1.)
            tex.select()
            gl.glDrawArrays(gl.GL_QUADS, 0, count)
            tex.deselect()

            ab.unbind()

        return geo, lfn

    def render(self, textData, **kw):
        lfn = self.layout(textData, **kw)[-1]
        return lfn()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def layoutText(self, textData, **kw):
        geo = textData.geometry.copy()
        geov = geo['v']

        align = self.align
        if align:
            wrapSize = kw['wrapSize']
            off = array([align*wrapSize, 0., 0.])
            for sl, width, placement in self.iterPlacements(textData, **kw):
                geov[sl] += placement + (off - align*width).round()
        else:
            for sl, width, placement in self.iterPlacements(textData, **kw):
                geov[sl] += placement

        return geo
    
    def iterPlacements(self, textData, **kw):
        offset = textData.getOffsetAtStart()

        for sl, width in self.wrapSlices(textData, **kw):
            off = offset[sl]
            if len(off):
                yield sl, width, offset[sl]
    
    def wrapText(self, textData, **kw):
        return self.wrapper.wrapText(textData, **kw)

    def wrapSlices(self, textData, **kw):
        return self.wrapper.wrapSlices(textData, **kw)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineWrapLayout(TextLayout):
    wrapper = textWrapping.LineTextWrapper()

    def __init__(self, wrapper=None):
        if wrapper is not None:
            self.wrapper = wrapper

    def iterPlacements(self, textData, wrapSlices=None, line=0, **kw):
        lineAdv = textData.lineAdvance
        offset = textData.getOffsetAtStart()
        for sl, width in self.wrapSlices(textData, **kw):
            off = offset[sl]
            if len(off):
                linePlacement = (line*lineAdv).round() - off[0]
                yield sl, width, (off + linePlacement)

            line += 1
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapLayout(LineWrapLayout):
    wrapper = textWrapping.FontTextWrapper()


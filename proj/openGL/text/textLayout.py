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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Layout
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextLayout(object):
    align = 0
    wrapper = textWrapping.BasicTextWrapper()
    
    def leftLayout(self, textObj, textData):
        geo = textData.geometry
        geov = geo['v']

        for sl, width, placement in self.iterPlacements(textObj, textData):
            geov[sl] += placement

        return geo
    __call__ = leftLayout

    def centerLayout(self, textObj, textData):
        geo = textData.geometry
        geov = geo['v']

        for sl, width, placement in self.iterPlacements(textObj, textData):
            geov[sl] += placement + (width*.5).round()

        return geo

    def rightLayout(self, textObj, textData):
        geo = textData.geometry
        geov = geo['v']

        for sl, width, placement in self.iterPlacements(textObj, textData):
            geov[sl] += placement + (width).round()

        return geo

    def alignLayout(self, textObj, textData):
        geo = textData.geometry
        geov = geo['v']

        align = textObj.align
        wrapSize = textObj.wrapSize
        alignOffset = array([align*wrapSize, 0., 0.])

        for sl, width, placement in self.iterPlacements(textObj, textData):
            geov[sl] += placement + (alignOffset - align*width).round()

        return geo
    
    def iterPlacements(self, textObj, textData):
        offset = textData.getOffsetAtStart()

        for sl, width in self.wrapSlices(textObj, textData):
            off = offset[sl]
            if len(off):
                yield sl, width, offset[sl]
    
    def wrapText(self, textObj, textData):
        return self.wrapper.wrapText(textObj, textData)

    def wrapSlices(self, textObj, textData):
        return self.wrapper.wrapSlices(textObj, textData)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineWrapLayout(TextLayout):
    wrapper = textWrapping.LineTextWrapper()

    def __init__(self, wrapper=None):
        if wrapper is not None:
            self.wrapper = wrapper

    def iterPlacements(self, textObj, textData):
        line = textObj.line
        lineAdv = textData.lineAdvance
        offset = textData.getOffsetAtStart()
        for sl, width in self.wrapSlices(textObj, textData):
            off = offset[sl]
            if len(off):
                linePlacement = (line*lineAdv).round() - off[0]
                yield sl, width, (off + linePlacement)

            line += 1
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapLayout(LineWrapLayout):
    wrapper = textWrapping.FontTextWrapper()


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
from . import textWrapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Layout
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextLayout(object):
    def layout(self, textObj, textData):
        wrapper = textObj.wrapper
        align = textObj.align
        width, height = textObj.size

        line = textObj.line
        #if width:
        #    alignOffset = array([align*width, 0., 0.])
        #else:
        #    alignOffset = array([0, 0., 0.])

        lineAdvance = textData.lineAdvance
        if height is not None:
            height = -height

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        geo = textData.geometry
        geov = geo['v']

        wrapSlices = wrapper.wrapSlices(textObj, textData)

        lineOffset = (line*lineAdvance).round()
        for textSlice, textOffset in wrapSlices:
            textOffset = textOffset + (lineOffset - textOffset[0])
            #textOffset[:,0] -= align * textOffset[-1][0]

            geov[textSlice] += textOffset[:-1]

            line += 1

            lineOffset = (line*lineAdvance).round()

            if lineOffset[1] < height: 
                return geo[:textSlice.stop]

        return geo


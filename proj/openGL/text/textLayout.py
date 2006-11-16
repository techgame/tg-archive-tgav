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
        align = textObj.align
        oneMinusAlign = 1-align

        pos = textData.AdvanceItem(textObj.pos)
        size = textData.AdvanceItem(textObj.size)
        size[0] *= align
        linePos = pos + size

        lineAdvance = textData.lineAdvance * textObj.lineSpacing

        # grab the geometry we are laying out
        geo = textData.geometry
        geov = geo['v']

        # ask for our slices
        wrapSlices = textObj.wrapper.wrapSlices(textObj, textData)

        # now layout
        if textObj.line:
            linePos -= (textObj.line*lineAdvance)

        for textSlice, textOffset in wrapSlices:
            alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
            lineOffset = (linePos - alignOff).round()
            geov[textSlice] += textOffset[:-1] + lineOffset

            linePos -= lineAdvance
            if (linePos<=pos)[1]:
                return geo[:textSlice.stop]

        return geo


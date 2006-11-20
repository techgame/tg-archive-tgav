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
        roundValues = textObj.roundValues
        crop = textObj.crop
        align = textObj.align[textObj.wrapAxis]
        oneMinusAlign = 1-align

        pos = textData.AdvanceItem(textObj.pos)
        size = textData.AdvanceItem(textObj.size)
        size[0] *= align
        linePos = pos + size

        lineAdvance = textData.lineAdvance * textObj.lineSpacing

        # grab the geometry we are laying out
        geo = textData.geometry.copy()
        geov = geo['v']

        # ask for our slices
        wrapSlices = textObj.wrapper.wrapSlices(textObj, textData)

        # now layout
        if textObj.line:
            linePos -= (textObj.line*lineAdvance)

        if roundValues:
            def getOffsetFor(textSlice, textOffset):
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff).round()
        else:
            def getOffsetFor(textSlice, textOffset):
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff)

        if crop:
            for textSlice, textOffset in wrapSlices:
                geov[textSlice] += getOffsetFor(textSlice, textOffset)
                linePos -= lineAdvance
                if (linePos<=pos)[1]:
                    return geo[:textSlice.stop]
        else:
            for textSlice, textOffset in wrapSlices:
                geov[textSlice] += getOffsetFor(textSlice, textOffset)
                linePos -= lineAdvance

        return geo


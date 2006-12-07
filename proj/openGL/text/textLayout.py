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

        pos = textData.AdvanceArray(textObj.pos).copy()
        size = textData.AdvanceArray(textObj.size).copy()

        linePos = size.copy()
        linePos[0] *= align
        linePos += pos

        lineAdvance = textData.lineAdvance * textObj.lineSpacing

        # grab the geometry we are laying out
        geo = textData.geometry.copy()
        geov = geo.v

        # ask for our slices
        wrapSlices = textObj.wrapper.wrapSlices(textObj, textData)

        # offset by lines (usually 1 or 0)
        if textObj.line:
            linePos -= (textObj.line*lineAdvance)

        # define some methods to handle alignment and offset
        if roundValues:
            def getOffsetFor(textOffset):
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff).round()
        else:
            def getOffsetFor(textOffset):
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff)

        lineCount = 0
        if crop:
            for textSlice, textOffset in wrapSlices:
                lineCount += 1
                geov[textSlice] += getOffsetFor(textOffset)
                linePos -= lineAdvance
                if (linePos<=pos)[1]:
                    geo = geo[:textSlice.stop]
                    break
        else:
            for textSlice, textOffset in wrapSlices:
                lineCount += 1
                geov[textSlice] += getOffsetFor(textOffset)
                linePos -= lineAdvance

        return geo


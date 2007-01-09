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

from numpy import array, zeros, concatenate
from . import textWrapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Layout
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextLayout(object):
    def layoutMesh(self, textObj, textData, wrapSlices):
        roundValues = textObj.roundValues
        crop = textObj.crop
        align = textObj.align[textObj.wrapAxis]
        oneMinusAlign = 1-align

        pos = textObj.box.pos
        size = textObj.box.size

        linePos = zeros(3, 'f')
        linePos[:2] = size*(align, 1) + pos

        lineAdvance = zeros(3, 'f')
        lineAdvance[:2] = textData.lineAdvance * textObj.lineSpacing
        nLinesCrop = int(size[1] // lineAdvance[1]) if crop else None

        # offset by lines (usually 1 or 0)
        if textObj.line:
            linePos -= (textObj.line*lineAdvance)

        # define some methods to handle alignment and offset
        offset = textData.getOffset()
        if roundValues:
            def getOffsetFor(textSlice):
                textOffset = offset[textSlice.start:textSlice.stop+1]
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff).round()
        else:
            def getOffsetFor(textSlice):
                textOffset = offset[textSlice.start:textSlice.stop+1]
                alignOff = (oneMinusAlign*textOffset[0] + align*textOffset[-1])
                return textOffset[:-1] + (linePos - alignOff)

        result = []
        # grab the geometry we are laying out
        geometry = textData.geometry.copy()
        textSlice = slice(0,0)
        for textSlice in wrapSlices[:nLinesCrop]:
            lgeom = geometry[textSlice]
            lgeom.v += getOffsetFor(textSlice)
            linePos -= lineAdvance
        geometry = geometry[:textSlice.stop]
        return geometry



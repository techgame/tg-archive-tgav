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

from itertools import izip, count
from numpy import array, zeros, concatenate
from . import textWrapping

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Layout
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextLayout(object):
    def layoutMesh(self, textObj, textData, iterWrapSlices):
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
        if crop:
            nLinesCrop = int(size[1] // lineAdvance[1])
            iterLineSlices = izip(iterWrapSlices, xrange(nLinesCrop))
        else:
            iterLineSlices = izip(iterWrapSlices, count())

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

        result = []
        # grab the geometry we are laying out
        geometry = textData.geometry.copy()
        offset = textData.getOffset()
        textSlice = slice(0,0)
        for textSlice, nline in iterLineSlices:
            textOffset = offset[textSlice.start:textSlice.stop+1]
            lgeom = geometry[textSlice]
            lgeom.v += getOffsetFor(textOffset)
            linePos -= lineAdvance
        geometry = geometry[:textSlice.stop]
        return geometry



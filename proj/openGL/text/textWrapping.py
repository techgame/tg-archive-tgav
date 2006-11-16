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

import re
from numpy import zeros_like

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Wrapping
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BasicTextWrapper(object):
    def wrapText(self, textObj, textData):
        text = textData.text
        for textSlice, textOffset in self.wrapSlices(textObj, textData):
            yield text[textSlice]

    def wrapSlices(self, textObj, textData):
        if not textData: return

        offset = textData.getOffset()
        for textSlice in self.availTextSlices(textData.text):
            textOffset = offset[textSlice.start:textSlice.stop+1]
            yield textSlice, textOffset

    def availTextSlices(self, text):
        if text: 
            return [slice(0, len(text))]
        else:
            return []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RETextWrapper(BasicTextWrapper):
    re_wrapPoints = re.compile('$|\n')

    def availTextSlices(self, text):
        if not text: return

        iterMatches = self.re_wrapPoints.finditer(text)
        i0 = 0
        for match in iterMatches:
            i1 = match.end()
            yield slice(i0, i1)
            i0 = i1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineTextWrapper(RETextWrapper):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapper(RETextWrapper):
    lineWraps = '\n'
    re_wrapPoints = re.compile('[\s-]|$')

    def wrapSlices(self, textObj, textData):
        wrapAxis = textObj.wrapAxis
        wrapSize = textObj.size[wrapAxis]
        if wrapSize <= 0: return

        text = textData.text
        if not text: return
        offset = textData.getOffset()

        lineWraps = self.lineWraps

        iLine = 0; offLine = offset[iLine, 0, wrapAxis]
        iCurr = iLine; offCurr = offLine
        for textSlice in self.availTextSlices(text):
            iNext = textSlice.stop
            offNext = offset[iNext, 0, wrapAxis]

            # check to see if the next wrap slice falls off the end
            if (wrapSize < (offNext - offLine)):
                yield (slice(iLine, iCurr), offset[iLine:iCurr+1])
                iLine = iCurr; offLine = offCurr

            # check to see if we have a linewrap at the current position
            if text[iNext-1] in lineWraps:
                yield (slice(iLine, iNext), offset[iLine:iNext+1])
                iLine = iNext; offLine = offNext

            iCurr = iNext; offCurr = offNext

        iNext = len(text)
        if iLine < iNext:
            yield (slice(iLine, iNext), offset[iLine:iNext+1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wrap Mode Map
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wrapModeMap = {
    'basic': BasicTextWrapper(),
    'line': LineTextWrapper(),
    'text': TextWrapper(),
    }
wrapModeMap[None] = wrapModeMap['basic']


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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Text Wrapping
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BasicTextWrapper(object):
    def iterWrapText(self, textObj, textData):
        text = textData.text
        for textSlice in self.wrapSlices(textObj, textData):
            yield text[textSlice]

    def iterWrapSlices(self, textObj, textData):
        if not textData: return

        offset = textData.getOffset()
        for textSlice in self.iterAvailTextSlices(textObj, textData):
            yield textSlice

    def iterAvailTextSlices(self, textObj, textData):
        text = textData.text
        if text: 
            return [slice(0, len(text))]
        else: return []

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RETextWrapper(BasicTextWrapper):
    re_wrapPoints = re.compile('$|\n\r')

    def iterAvailTextSlices(self, textObj, textData):
        text = textData.text
        if not text: return

        iterMatches = self.re_wrapPoints.finditer(text)
        i0 = 0
        for match in iterMatches:
            i1 = match.end()
            yield slice(i0, i1)
            i0 = i1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineTextWrapper(RETextWrapper):
    re_wrapPoints = re.compile('$|\n\r')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapper(RETextWrapper):
    lineWraps = '\n\r'
    re_wrapPoints = re.compile('[\s-]|$')

    def iterWrapSlices(self, textObj, textData):
        wrapAxis = textObj.wrapAxis
        wrapSize = textObj.box.size[wrapAxis]
        if wrapSize <= 0: return

        text = textData.text
        if not text: return
        offset = textData.getOffset()

        lineWraps = self.lineWraps

        iLine = 0; offLine = offset[iLine, 0, wrapAxis]
        iCurr = iLine; offCurr = offLine
        for textSlice in self.iterAvailTextSlices(textObj, textData):
            iNext = textSlice.stop
            offNext = offset[iNext, 0, wrapAxis]

            # check to see if the next wrap slice falls off the end
            if (wrapSize < (offNext - offLine)):
                yield slice(iLine, iCurr)
                iLine = iCurr; offLine = offCurr

            # check to see if we have a linewrap at the current position
            if text[iNext-1] in lineWraps:
                yield slice(iLine, iNext)
                iLine = iNext; offLine = offNext

            iCurr = iNext; offCurr = offNext

        iNext = len(text)
        if iLine < iNext:
            yield slice(iLine, iNext)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wrap Mode Map
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wrapModeMap = {
    'basic': BasicTextWrapper(),
    'line': LineTextWrapper(),
    'text': TextWrapper(),
    }
wrapModeMap[None] = wrapModeMap['basic']


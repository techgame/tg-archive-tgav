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
        for sl, width in self.wrapSlices(textObj, textData):
            yield text[sl]

    def wrapSlices(self, textObj, textData):
        yield slice(None), textData.getOffset()[-1,0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineTextWrapper(BasicTextWrapper):
    re_wrapPoints = re.compile('\n')

    def _iterWrapIndexes(self, textObj, textData):
        return ((m.start(), m.group()) for m in self.re_wrapPoints.finditer(textData.text))

    def wrapSlices(self, textObj, textData):
        iterWrapIdx = self._iterWrapIndexes(textObj, textData)

        offsetAtEnd = textData.getOffsetAtEnd()
        if not len(offsetAtEnd):
            return

        i0 = 0
        lineOffset = zeros_like(offsetAtEnd[0])
        for iCurr, subtext in iterWrapIdx:
            newLineOffset = offsetAtEnd[iCurr]
            yield slice(i0, iCurr+1), (newLineOffset - lineOffset)[0]
            i0 = iCurr+1
            lineOffset = newLineOffset
    
        iEnd = len(offsetAtEnd)-1
        if i0 > iEnd:
            return
    
        newLineOffset = offsetAtEnd[-1]
        yield slice(i0, None), (newLineOffset - lineOffset)[0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FontTextWrapper(LineTextWrapper):
    lineWrapSet = set(['\n'])
    re_wrapPoints = re.compile('[\s-]')

    def wrapSlices(self, textObj, textData):
        iterWrapIdx = self._iterWrapIndexes(textObj, textData)

        offsetAtEnd = textData.getOffsetAtEnd()
        if not len(offsetAtEnd):
            return

        wrapSize = textObj.wrapSize
        wrapAxis = textObj.wrapAxis
        lineWrapSet = self.lineWrapSet

        lineOffset = zeros_like(offsetAtEnd[0])
        i0 = 0
        iPrev = 0
        prevOffsetAtEnd = lineOffset
        for iCurr, subtext in iterWrapIdx:
            currOffsetAtEnd = offsetAtEnd[iCurr]

            # check to see if the current word falls off the end
            if wrapSize < (currOffsetAtEnd - lineOffset)[0, wrapAxis]:
                # it does, so wrap to the previous wrap point
                newLineOffset = prevOffsetAtEnd
                yield slice(i0, iPrev+1), (newLineOffset - lineOffset)[0]
                i0 = iPrev+1
                lineOffset = newLineOffset

            # check to see if we have a linewrap at the current position
            if subtext in lineWrapSet:
                # there is one, so wrap here
                newLineOffset = currOffsetAtEnd
                yield slice(i0, iCurr+1), (newLineOffset - lineOffset)[0]
                i0 = iCurr+1
                lineOffset = newLineOffset

            iPrev = iCurr
            prevOffsetAtEnd = currOffsetAtEnd

        iEnd = len(offsetAtEnd)-1
        if i0 > iEnd:
            return

        # make sure the last line is wrapped properly
        if wrapSize < (offsetAtEnd[iEnd] - lineOffset)[0, wrapAxis]:
            newLineOffset = offsetAtEnd[iPrev]
            yield slice(i0, iPrev+1), (newLineOffset - lineOffset)[0]
            i0 = iPrev+1
            lineOffset = newLineOffset
            iPrev = iEnd

        newLineOffset = offsetAtEnd[-1]
        yield slice(i0, None), (newLineOffset - lineOffset)[0]


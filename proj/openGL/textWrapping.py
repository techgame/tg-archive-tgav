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
    def wrapText(self, textData, wrapSize=None):
        for sl, width in self.wrapSlices(textData, wrapSize):
            yield text[sl]

    def wrapSlices(self, textData, wrapSize=None):
        yield slice(None), textData.getOffset()[-1,0]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineTextWrapper(BasicTextWrapper):
    def wrapSlices(self, textData, wrapSize=None):
        iterWrapIdx = self._iterWrapIndexes(textData)
        return self._wrapSliceIndexes(iterWrapIdx, textData, wrapSize)

    re_wrapPoints = re.compile('\n')
    def _iterWrapIndexes(self, textData):
        return ((m.start(), m.group()) for m in self.re_wrapPoints.finditer(textData.text))

    def _wrapSliceIndexes(self, iterWrapIdx, textData, wrapSize=None):
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
    re_wrapPoints = re.compile('\s')

    def __init__(self, wrapSize=None, axis=0):
        self.wrapSize = wrapSize
        self.axis = 0

    def _wrapSliceIndexes(self, iterWrapIdx, textData, wrapSize=None):
        wrapSize = wrapSize or self.wrapSize
        lineWrapSet = self.lineWrapSet

        offsetAtEnd = textData.getOffsetAtEnd()
        if not len(offsetAtEnd):
            return

        lineOffset = zeros_like(offsetAtEnd[0])
        i0 = 0
        iPrev = 0
        for iCurr, subtext in iterWrapIdx:
            # force a feed if linewrap
            if subtext in lineWrapSet:
                newLineOffset = offsetAtEnd[iCurr]
                yield slice(i0, iCurr+1), (newLineOffset - lineOffset)[0]
                i0 = iCurr+1
                lineOffset = newLineOffset
                iPrev = iCurr

            elif wrapSize < (offsetAtEnd[iCurr] - lineOffset)[0, self.axis]:
                newLineOffset = offsetAtEnd[iPrev]
                yield slice(i0, iPrev+1), (newLineOffset - lineOffset)[0]
                i0 = iPrev+1
                lineOffset = newLineOffset
                iPrev = iCurr

            else:
                iPrev = iCurr

        iEnd = len(offsetAtEnd)-1
        if i0 > iEnd:
            return

        # make sure the last line is wrapped properly
        if wrapSize < (offsetAtEnd[iEnd] - lineOffset)[0,self.axis]:
            newLineOffset = offsetAtEnd[iPrev]
            yield slice(i0, iPrev+1), (newLineOffset - lineOffset)[0]
            i0 = iPrev+1
            lineOffset = newLineOffset
            iPrev = iEnd

        newLineOffset = offsetAtEnd[-1]
        yield slice(i0, None), (newLineOffset - lineOffset)[0]


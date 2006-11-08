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
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ITextLayout(object):
    def layout(self, text):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

class SimpleTextLayout(ITextLayout):
    def layout(self, text):
        pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrap(object):
    lineWrapSet = set(['\n'])
    re_wrapPoints = re.compile('\s')

    def __init__(self, wrapSize, axis=0):
        self.wrapSize = wrapSize
        self.axis = 0

    def wrap(self, textObj):
        text = textObj.text
        for s in self._sliceByText(textObj):
            yield text[i0:i1]

    def _sliceByText(self, textObj):
        iterWrapIdx = self._iterWrapIndexes(textObj)
        return self._sliceByIndex(iterWrapIdx, textObj)

    def _iterWrapIndexes(self, textObj):
        return ((m.start(), m.group()) for m in self.re_wrapPoints.finditer(textObj.text))

    def _sliceByIndex(self, iterWrapIdx, textObj):
        wrapSize = self.wrapSize
        lineWrapSet = self.lineWrapSet

        advSum = textObj.advance[:, 0, self.axis].cumsum(0)

        lineOffset = zeros_like(advSum[0])
        i0 = 0
        iPrev = None
        for iCurr, subtext in iterWrapIdx:
            if wrapSize < (advSum[iCurr] - lineOffset):
                yield slice(i0, iPrev+1)
                i0 = iPrev+1
                lineOffset = advSum[iPrev]
                iPrev = iCurr

            else:
                iPrev = iCurr

            # force a feed if linewrap
            if subtext in lineWrapSet:
                yield slice(i0, iCurr+1)
                i0 = iCurr+1
                lineOffset = advSum[iCurr]
                iPrev = iCurr

        iEnd = len(advSum)-1
        if i0 > iEnd:
            return

        # make sure the last line is wrapped properly
        if wrapSize < (advSum[iEnd] - lineOffset):
            yield slice(i0, iPrev+1)
            i0 = iPrev+1
            lineOffset = advSum[iPrev]
            iPrev = iEnd

        yield slice(i0, None)


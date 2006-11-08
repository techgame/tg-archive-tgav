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

class TextWrapper(object):
    lineWrapSet = set(['\n'])
    re_wrapPoints = re.compile('\s')

    def __init__(self, wrapSize=None, axis=0):
        self.wrapSize = wrapSize
        self.axis = 0

    def wrap(self, textObj, wrapSize=None):
        wrapSize = wrapSize or self.wrapSize
        text = textObj.text
        for s in self._sliceByText(textObj, wrapSize):
            yield text[s]

    def _sliceByText(self, textObj, wrapSize):
        iterWrapIdx = self._iterWrapIndexes(textObj)
        return self._sliceByIndex(iterWrapIdx, textObj, wrapSize)

    def _iterWrapIndexes(self, textObj):
        return ((m.start(), m.group()) for m in self.re_wrapPoints.finditer(textObj.text))

    def _sliceByIndex(self, iterWrapIdx, textObj, wrapSize):
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextObject(object):
    text = ""

    font = None
    texture = None

    wrapper = TextWrapper()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def factoryFor(klass, font):
        subklass = type(klass)(klass.__name__+'_T_', (klass,), {})
        subklass.setupClassFont(font)
        return subklass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, text, font=None):
        self.text = text

        if font:
            self.setupFont(self, font)

    def setupFont(self, font):
        self.font = font
        self.texture = font.texture

    setupClassFont = classmethod(setupFont)

    _text = None
    def getText(self):
        return self._text
    def setText(self, text):
        self._text = text
        self._recache()
    text = property(getText, setText)

    def _recache(self):
        self.idx = self.font.translate(self._text)
        self._advance = None
        self._geometry = None

    _advance = None
    def getAdvance(self):
        r = self._advance
        if r is None:
            r = self.font.advance[self.idx]
            self._advance = r
        return r
    advance = property(getAdvance)

    _geometry = None
    def getGeometry(self):
        r = self._geometry
        if r is None:
            r = self.font.geometry[self.idx]
            self._geometry = r
        return r
    geometry = property(getGeometry)

    def wrap(self, wrapSize=None):
        return self.wrapper.wrap(self, wrapSize)


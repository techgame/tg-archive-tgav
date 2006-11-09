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
from numpy import zeros_like, asarray

from TG.observing import Observable
from TG.openGL.raw import gl
from TG.openGL.bufferObjects import ArrayBuffer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextObject(Observable):
    font = None
    texture = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def factoryFor(klass, font):
        subklass = type(klass)(klass.__name__+'_T_', (klass,), {})
        subklass.setupClassFont(font)
        return subklass

    def setupFont(self, font):
        self.font = font
        self.texture = font.texture
    setupClassFont = classmethod(setupFont)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, text, font=None):
        if font is not None:
            self.setupFont(font)
        self.text = text

    text = Observable.obproperty('')
    @text.fset
    def setText(self, text, _obSet_):
        _obSet_(text)
        self._recache()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _recache(self):
        self.srcidx = self.font.translate(self.text)
        self._advance = None
        self._geometry = None
        self._offset = None

    _advance = None
    def getAdvance(self, startAtNull=False):
        r = self._advance
        if r is None:
            r = self.font.advance[[0]+self.srcidx]
            self._advance = r
            self.lineAdvance = asarray(self.font.lineAdvance, dtype=r.dtype)

        if startAtNull: return r
        else: return r[1:]
    def getPreAdvance(self):
        return self.getAdvance(True)[:-1]
    def getPostAdvance(self):
        return self.getAdvance(False)

    advance = property(getAdvance)

    _offset = None
    def getOffset(self, startAtNull=False):
        r = self._offset
        if r is None:
            r = self.getAdvance(True).cumsum(0)
            self._offset = r

        if startAtNull: return r
        else: return r[1:]

    def getOffsetAtStart(self):
        return self.getOffset(True)[:-1]
    def getOffsetAtEnd(self):
        return self.getOffset(False)

    offset = property(getOffset)

    _geometry = None
    def getGeometry(self):
        r = self._geometry
        if r is None:
            r = self.font.geometry[self.srcidx]
            self._geometry = r
        return r
    geometry = property(getGeometry)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapper(object):
    lineWrapSet = set(['\n', '\r'])
    re_wrapPoints = re.compile('\s')

    def __init__(self, wrapSize=None, axis=0):
        self.wrapSize = wrapSize
        self.axis = 0

    def wrap(self, textObj, wrapSize=None):
        for sl, sloff in self.wrapSlices(textObj, wrapSize):
            yield text[sl]

    def wrapSlices(self, textObj, wrapSize=None):
        iterWrapIdx = self._iterWrapIndexes(textObj)
        return self.wrapSliceIndexes(iterWrapIdx, textObj, wrapSize)

    def _iterWrapIndexes(self, textObj):
        return ((m.start(), m.group()) for m in self.re_wrapPoints.finditer(textObj.text))

    def wrapSliceIndexes(self, iterWrapIdx, textObj, wrapSize=None):
        wrapSize = wrapSize or self.wrapSize
        lineWrapSet = self.lineWrapSet

        offsetAtEnd = textObj.getOffsetAtEnd()[:, :]
        if not len(offsetAtEnd):
            return

        lineOffset = zeros_like(offsetAtEnd[0])
        i0 = 0
        iPrev = 0
        for iCurr, subtext in iterWrapIdx:
            # force a feed if linewrap
            if subtext in lineWrapSet:
                newLineOffset = offsetAtEnd[iCurr]
                yield slice(i0, iCurr+1), newLineOffset - lineOffset
                i0 = iCurr+1
                lineOffset = newLineOffset
                iPrev = iCurr

            elif wrapSize < (offsetAtEnd[iCurr] - lineOffset)[0, self.axis]:
                newLineOffset = offsetAtEnd[iPrev]
                yield slice(i0, iPrev+1), newLineOffset - lineOffset
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
            yield slice(i0, iPrev+1), newLineOffset - lineOffset
            i0 = iPrev+1
            lineOffset = newLineOffset
            iPrev = iEnd

        newLineOffset = offsetAtEnd[-1]
        yield slice(i0, None), newLineOffset - lineOffset

wrapper = TextWrapper()
wrap = wrapper.wrap

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextWrapLayout(Observable):
    def __init__(self, textWrapper=wrapper):
        self.textWrapper = textWrapper

    def layoutGeometry(self, textObj, wrapSize=None, line=0):
        geo = textObj.geometry.copy()

        offset = textObj.getOffsetAtStart()
        advance = textObj.getPostAdvance()

        lineAdv = textObj.lineAdvance
        for sl, sloff in self.textWrapper.wrapSlices(textObj, wrapSize):
            off = offset[sl]
            if not len(off):
                line += 1
                continue

            off = off - off[0]

            adv = advance[sl]
            width = off[-1][0] + adv[-1][0]

            if 1: # left
                width[:] = 0
            elif 1: # center
                width[0] = 0.5*(wrapSize-width[0])
            elif 1: # right
                width[0] = (wrapSize-width[0])

            geo['v'][sl] += (off + ((line*lineAdv) + width)).round()#- off[0])

            line += 1
        return geo

    def layout(self, textObj, wrapSize=None, line=0):
        geo = self.layoutGeometry(textObj, wrapSize, line)
        def lfn(tex=textObj.texture, geo=geo):
            tex.select()
            geo.draw(gl.GL_QUADS)
        return geo, lfn

    def layoutBuffer(self, textObj, wrapSize=None, line=0):
        geo = self.layoutGeometry(textObj, wrapSize, line)

        ab = ArrayBuffer()
        ab.sendData(geo)
        def lfn(tex=textObj.texture, ab=ab, dataFormat=geo.dataFormat, count=len(geo.flat)):
            tex.select()
            ab.bind()
            gl.glInterleavedArrays(dataFormat, 0, 0)
            gl.glDrawArrays(gl.GL_QUADS, 0, count)
            ab.unbind()
        return geo, lfn

    def render(self, textObj, wrapSize=None):
        geo, lfn = self.layout(textObj, wrapSize)
        lfn()


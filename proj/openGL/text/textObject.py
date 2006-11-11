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

from . import textLayout
from . import textRenderer

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextObject(object):
    UnbufferedDisplayFactory = textRenderer.TextDisplay
    BufferedDisplayFactory = textRenderer.TextBufferedDisplay

    display = None
    layout = textLayout.TextLayout()
    textData = None

    line = 1
    wrapSize = 0
    wrapAxis = 0
    align = 0

    def __init__(self, textData=None):
        self.textData = textData
        self.buffered = self.buffered

    def setFromFont(self, font):
        self.textData = font.textData(self.getText())

    @classmethod
    def fromFont(klass, font, text=''):
        textData = font.textData(text)
        return klass(textData)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _text = ''
    def getText(self):
        return self._text
    def setText(self, text, doUpdate=False):
        self._text = text
        if doUpdate:
            self.update(text)
    text = property(getText, setText)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def wrapMode(self, mode=None):
        if mode == None:
            self.layout = None
            del self.layout
        elif mode == 'none':
            self.layout = textLayout.TextLayout()
        elif mode == 'line':
            self.layout = textLayout.LineWrapLayout()
        elif mode == 'text':
            self.layout = textLayout.TextWrapLayout()
        else:
            raise ValueError("Mode %r was not found" % (mode,))

        self.update()

    _buffered = False
    def getBuffered(self):
        return self._buffered
    def setBuffered(self, buffered=True):
        self._buffered = buffered
        if buffered:
            display = self.BufferedDisplayFactory()
        else:
            display = self.UnbufferedDisplayFactory()
        self.display = display
        self.update()
    def setUnbuffered(self, unbuffered=True):
        self.setBuffered(not unbuffered)
    buffered = property(getBuffered, setBuffered)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self, text=None):
        textData = self.textData
        if textData is None:
            return False

        if text is not None:
            self._text = text
            textData.text = self._text

        geo = self.layout(self, textData)
        self.display.update(self, textData, geo)
        return True

    def __call__(self):
        self.display()


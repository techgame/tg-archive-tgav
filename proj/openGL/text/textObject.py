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

    def __init__(self, **kwattr):
        self.set(kwattr)

    def set(self, val=None, **kwattr):
        for n,v in (val or kwattr).iteritems():
            setattr(self, n, v)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getFont(self, font):
        return self.textData.font
    def setFont(self, font):
        self.textData = font.textData(self.getText())
    font = property(getFont, setFont)

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

    def setWrapMode(self, mode=None):
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
    wrapMode = property(fset=setWrapMode)

    _buffered = True
    def getBuffered(self):
        return self._buffered
    def setBuffered(self, buffered=True):
        self._buffered = buffered
        self.display = None
    def setUnbuffered(self, unbuffered=True):
        self.setBuffered(not unbuffered)
    buffered = property(getBuffered, setBuffered)

    _display = None
    def getDisplay(self):
        display = self._display
        if display is None:
            if self.buffered:
                display = self.BufferedDisplayFactory()
            else:
                display = self.UnbufferedDisplayFactory()
            self._display = display
            self.update()
        return display
    def setDisplay(self, display):
        self._display = display
    display = property(getDisplay, setDisplay)

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


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
from . import textWrapping
from ..shapes import PositionalObject

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextObject(PositionalObject):
    layout = textLayout.TextLayout()

    WrapModeMap = textWrapping.wrapModeMap
    wrapper = WrapModeMap[None]

    DisplayFactoryMap = textRenderer.displayFactoryMap
    DisplayFactory = DisplayFactoryMap[None]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    textData = None

    line = 1
    lineSpacing = 1
    crop = True
    wrapAxis = 0

    def __init__(self, text=None, **kwattr):
        self.text = text
        self.set(kwattr)
        self.update()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getFont(self):
        return self.textData.font
    def setFont(self, font):
        if self.textData:
            self.textData.recompile()
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

    def setWrapMode(self, mode=None, doUpdate=False):
        self.wrapper = self.WrapModeMap[mode]
        if doUpdate:
            self.update()
    wrapMode = property(fset=setWrapMode)

    def setDisplayMode(self, mode=None):
        self.DisplayFactory = self.DisplayFactoryMap[mode]
    displayMode = property(fset=setWrapMode)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _display = None
    def getDisplay(self):
        display = self._display
        if display is None:
            display = self.DisplayFactory()
            self._display = display
        return display
    def setDisplay(self, display, doUpdate=False):
        self._display = display
        if doUpdate:
            self.update()
    display = property(getDisplay, setDisplay)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def update(self, text=None, **kwattr):
        if kwattr: 
            self.set(kwattr)

        textData = self.textData
        if textData is None:
            return False

        if text is not None:
            self._text = text

        textData.text = self.text
        geo = self.layout.layout(self, textData)
        self.display.update(self, textData, geo)
        return True

    def render(self):
        self.display.render()


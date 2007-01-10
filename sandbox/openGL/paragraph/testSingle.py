##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~ Copyright (C) 2002-2004  TechGame Networks, LLC.
##~ 
##~ This library is free software; you can redistribute it and/or
##~ modify it under the terms of the BSD style License as found in the 
##~ LICENSE file included with this distribution.
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from __future__ import with_statement
import string
import time

from renderBase import RenderSkinModelBase

from TG.openGL.text import Font

from TG.openGL import glBlock, glMatrix

from TG.openGL.raw import gl, glu, glext
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

from textObject import TextObject

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    sampleText = "FreeType 2\nAnd another line\nThird line"
    fontName, fontSize = 'AndaleMono', 80

    fps = 60
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
            'Courier': '/System/Library/Fonts/Courier.dfont',
            'CourierNew': '/Library/Fonts/Courier New',
            'AndaleMono': '/Library/Fonts/Andale Mono',
            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',

            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'Herculanum': '/Library/Fonts/Herculanum.dfont',
            'Papyrus': '/Library/Fonts/Papyrus.dfont',

            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT',
            'AmericanTypewriter': '/Library/Fonts/AmericanTypewriter.dfont',
            'Helvetica': '/System/Library/Fonts/Helvetica.dfont',
            }

    clearMask = GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT
    def renderInit(self, glCanvas, renderStart):
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(0.15, 0.15, 0.25, 1.)
        glClear(self.clearMask)

        self.contentText = TextObject(text=self.sampleText, font=self.loadFont(self.fontName, self.fontSize), color='#bbf', size=(600.600))
        self.refreshText(False)

    def refreshText(self, bRefresh=True, **kw):
        self.contentText.update(**kw)
        if bRefresh and self.fps <= 0:
            self.canvas.Refresh()

    def refreshFont(self, bRefresh=True):
        self.contentText.font.size = self.fontSize

        if bRefresh:
            self.refreshText(bRefresh)

    def loadFont(self, fontKey, fontSize, dpi=None, charset=string.printable):
        fontFilename = self.fonts[fontKey]
        f = Font.fromFilename(fontFilename, fontSize, dpi=dpi, charset=charset)
        return f

    def onChar(self, evt):
        ch = unichr(evt.GetUniChar()).replace('\r', '\n')

        if ch in ('-', '_'):
            self.fontSize = max(4, self.fontSize - 1)
            print 'dec', self.fontSize
            self.refreshFont()
        elif ch in ('+', '='):
            self.fontSize += 1
            print 'inc', self.fontSize
            self.refreshFont()
        elif ch in ('*', ):
            self.setFps(-self.fps)
        elif ch in ('\x7f',):
            # delete
            self.sampleText = ""
            self.refreshText(text=self.sampleText)
        elif ch in ('\x08',):
            # backspace
            self.sampleText = self.sampleText[:-1]
            self.refreshText(text=self.sampleText)
        elif ch in string.printable:
            self.sampleText += ch
            self.refreshText(text=self.sampleText)
        else:
            print repr(ch)
            self.sampleText += ch.encode("unicode_escape")
            self.refreshText(text=self.sampleText)

    viewPortSize = (800, 800)
    def renderResize(self, glCanvas):
        (w, h) = glCanvas.GetSize()
        if not w or not h: return

        border = 0
        (l, b), (w, h) = (border, border), (w-2*border, h-2*border)

        self.viewPortSize = w, h
        glViewport (l, b, w, h)

        self.contentText.box.size = w,h
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        glOrtho(0, w, 0, h, -100, 100)
        #glTranslatef(10, h-10, 0)

        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()

        self.refreshText(False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        glClear(self.clearMask)
        self.contentText.render()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    m = RenderSkinModel()
    m.skinModel()


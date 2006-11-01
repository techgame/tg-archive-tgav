#!/usr/bin/env python
#!/usr/local/bin/python2.5
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

import string
import time

from renderBase import RenderSkinModelBase

from TG.openGL.raw import gl, glu, glext
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

import fontTexture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
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

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True

    def renderInit(self, glCanvas, renderStart):
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(1., 1., 1.,1.)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        self.fontOne = self.loadFontTexture('AndaleMono', 32)
        self.fontTwo = self.loadFontTexture('Zapfino', 72)

    def loadFontTexture(self, fontKey, fontSize):
        fontFilename = self.fonts[fontKey]
        if 1:
            font = fontTexture.GLFreetypeFaceRect(fontFilename, fontSize)
        else:
            font = fontTexture.GLFreetypeFace2D(fontFilename, fontSize)
        font.loadChars(string.printable)
        #font.face.printInfo()
        return font

    viewPortSize = None
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        w, h = w-20, h-20
        self.viewPortSize = w, h
        glViewport (10, 10, w, h)

        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0, w, -h, 0)

        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    i = 0
    def renderContent(self, glCanvas, renderStart):
        if self.viewPortSize is None: return

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glClearColor(1., 1., 1., 1.)

        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)

        glLoadIdentity()

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(glext.GL_TEXTURE_RECTANGLE_ARB)

        x0 = y1 = 0
        x1, y0 = self.viewPortSize
        y0 = -y0

        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(1., 1., 1., 1.)
        gl.glVertex2f(x0, y1)
        gl.glColor4f(1., 0.5, 0.5, 1.)
        gl.glVertex2f(x0, y0)
        gl.glColor4f(0.5, 0.5, 1., 1.)
        gl.glVertex2f(x1, y0)
        gl.glColor4f(0.5, 1., 0.5, 1.)
        gl.glVertex2f(x1, y1)
        gl.glEnd()

        if 1:
            for e in self.fontOne.selectScale():
                for e in self.fontOne.select():
                    e.drawBreak()
                    glColor4f(0., 0., 0., 1.)
                    e.drawString(self.fpsStr)
                    e.drawBreak()
                    e.drawString("Shane was here!")

                for e in self.fontTwo.select():
                    e.drawBreak()

                    glColor4f(1., 1., 1., 1.)
                    e.drawString('render start: ' + time.asctime())
                    e.drawBreak()

                    self.i+=1
                    glColor4f(0., 0., 0., 1.)
                    e.drawString('loop count: %s' % (self.i, ))

    fpsStr = 'Waiting...'
    def _printFPS(self, fpsStr):
        self.fpsStr = fpsStr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


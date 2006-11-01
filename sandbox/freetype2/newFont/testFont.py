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

from TG.tgavUtility.renderBase import RenderSkinModelBase

from TG.openGL.raw import gl, glu, glext
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

from TG.openGL.font import Font

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fps = 1
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

        #self.fontOne = self.loadFont('AndaleMono', 72)
        self.fontOne = self.loadFont('Zapfino', 72)
        #self.fontTwo = self.loadFont('Zapfino', 72)

        geo = self.fontOne.layout("Four")# score, and seven years ago!")

    def loadFont(self, fontKey, fontSize):
        global f
        fontFilename = self.fonts[fontKey]
        f = Font.fromFilename(fontFilename, fontSize)
        f.configure()
        return f

    viewPortSize = None
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        w, h = w-20, h-20
        self.viewPortSize = w, h
        glViewport (10, 10, w, h)

        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0, w, 0, h)

        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    i = 0
    def renderContent(self, glCanvas, renderStart):
        if self.viewPortSize is None: return

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)

        glLoadIdentity()

        x0 = y0 = 0
        x1, y1 = self.viewPortSize

        if 1:
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
            glPushMatrix()
            glTranslatef(0., 100., .5)

            glTranslatef(3., -1., -.01)
            if 1: gl.glColor4f(0., 0., 0., 0.3)
            self.fontOne.render("Four score, and seven years ago!")

            glTranslatef(-3., 1., +.02)

            if 1: gl.glColor4f(1., 1., 1., 1.)
            elif 1: gl.glColor4f(0., 0., 0., 1.0)
            self.fontOne.render("Four score, and seven years ago!")

            glPopMatrix()

        if 0:
            tex = self.fontOne.texture
            s0 = 0
            s1 = tex.size[0]
            t0 = 0
            t1 = tex.size[1]
            tex.select()

            #print (x0, y0, x1, y1), (s0, t0, s1, t1)
            #(x1-x0)/abs(s1-s0)

            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(0., 0., 0., 0.5)
            gl.glTexCoord2s(s0, t1)
            gl.glVertex2f(x0, y0)

            gl.glColor4f(0., 0., 0., 0.5)
            gl.glTexCoord2s(s1, t1)
            gl.glVertex2f(x1, y0)

            gl.glColor4f(0., 0., 0., 0.5)
            gl.glTexCoord2s(s1, t0)
            gl.glVertex2f(x1, y1)

            gl.glColor4f(0., 0., 0., 0.5)
            gl.glTexCoord2s(s0, t0)
            gl.glVertex2f(x0, y1)
            gl.glEnd()

    fpsStr = 'Waiting...'
    def _printFPS(self, fpsStr):
        self.fpsStr = fpsStr

    def checkError(self):
        glerr = gl.glGetError()
        if glerr:
            raise Exception("GLError: %d (0x%x)" % (glerr, glerr))
        return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    m = RenderSkinModel()
    m.skinModel()


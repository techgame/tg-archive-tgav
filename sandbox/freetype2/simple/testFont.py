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
import platform

from renderBase import RenderSkinModelBase

from TG.openGL.raw import gl, glu, glext
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

from TG.openGL.font import Font

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fps = 60
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

        #glClearColor(1., 1., 1.,1.)
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        self.fontOne = self.loadFont('Arial', 48)
        self.fontTwo = self.loadFont('Zapfino', 72)
        self.fontThree = self.loadFont('AndaleMono', 18)

        global fpsGeo, fpsEnd, fpsRenderText, verGeo, verEnd, verRenderText, stuffGeo, stuffEnd, stuffText, openglGeo, openglEnd, openglRocksText
        verGeo, verEnd, verRenderText = self.fontThree.layout(platform.version())
        stuffGeo, stuffEnd, stuffText = self.fontTwo.layout(platform.node())
        openglGeo, openglEnd, openglRocksText = self.fontTwo.layout("OpenGL rocks!")

    def loadFont(self, fontKey, fontSize):
        global f
        fontFilename = self.fonts[fontKey]
        f = Font.fromFilename(fontFilename, fontSize)
        f.setCharset(string.printable)
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
        global fpsGeo, fpsEnd, fpsRenderText, verGeo, verEnd, verRenderText, stuffGeo, stuffEnd, stuffText, openglGeo, openglEnd, openglRocksText

        if self.viewPortSize is None: return

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_MODULATE)

        glLoadIdentity()

        x0 = y0 = 0
        x1, y1 = self.viewPortSize

        if 0:
            glTranslatef(0., 0., -1)
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
            glTranslatef(0., 0., 1)

        if 1:
            s = abs(1. - (renderStart % 2.0))
            a = 1.0 #+ 0.5*abs(1. - (renderStart % 2.0)) 

            if 1: gl.glColor4f(1., 1., 1., a)
            elif 1: gl.glColor4f(0., 0., 0., a)

            fpsGeo, fpsEnd, fpsRenderText = self.fontOne.layout(self.fpsStr)

            def centerEnd(end):
                glTranslatef(0.5*(self.viewPortSize[0]-end[0]), 0, 0)
            def rightEnd(end):
                glTranslatef((self.viewPortSize[0]-end[0]), 0, 0)

            def showText(rt):
                #glTranslatef(3., -1., -.1)
                #if 1: gl.glColor4f(0., 0., 0., 0.3)
                #rt()

                #glTranslatef(-3., 1., +.05)

                if 1: gl.glColor4f(1., 1., 1., a)
                elif 1: gl.glColor4f(0., 0., 0., a)
                rt()

            def nullify(f): 
                return None

            @nullify
            def showGeoOutline(geo):
                geov = geo['v']
                x0, y0, z0 = geov.min(0).min(0)
                x1, y1, z1 = geov.max(1).max(0)

                glColor4f(0., 0., 1., .5)
                glBegin(GL_QUADS)
                glVertex2f(x0, y0)
                glVertex2f(x1, y0)
                glVertex2f(x1, y1)
                glVertex2f(x0, y1)
                glEnd()

                glTranslatef(0., 0., .01)

            @nullify
            def showOutline(geo):
                glPushMatrix()
                glTranslatef(0., 0., -.05)
                if showGeoOutline is not None:
                    showGeoOutline(geo)
                glColor4f(0., 1., 0., .5)
                geo.draw(GL_QUADS)
                glPopMatrix()

            @nullify
            def showBaseline(end):
                glColor4f(1., 0., 0., .5)
                glBegin(GL_LINES)
                glVertex3f(0., 0., 0.)
                glVertex3f(*end)
                glEnd()

            glPushMatrix()
            glTranslatef(0, self.viewPortSize[1], 0)

            glTranslatef(0, -40, 0)
            if showOutline is not None:
                showOutline(fpsGeo)
            if showBaseline is not None:
                showBaseline(fpsEnd)
            showText(fpsRenderText)

            glTranslatef(0, -200, 0)
            glPushMatrix()
            centerEnd(stuffEnd)
            if showOutline is not None:
                showOutline(stuffGeo)
            if showBaseline is not None:
                showBaseline(stuffEnd)
            showText(stuffText)
            glPopMatrix()

            glTranslatef(0, -100, 0)
            glPushMatrix()
            #glScalef(2,2,1)
            rightEnd(verEnd)
            if showOutline is not None:
                showOutline(verGeo)
            if showBaseline is not None:
                showBaseline(verEnd)
            showText(verRenderText)
            glPopMatrix()

            glTranslatef(0, -200, 0)
            centerEnd(openglEnd)
            if showOutline is not None:
                showOutline(openglGeo)
            if showBaseline is not None:
                showBaseline(openglEnd)
            showText(openglRocksText)

            glPopMatrix()

        if 0:
            tex = self.fontOne.texture
            s0 = 0
            s1 = tex.size[0]
            t0 = 0
            t1 = tex.size[1]
            tex.select()

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


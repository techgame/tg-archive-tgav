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
import array

from ctypes import byref, c_void_p

from TG.tgavUtility.renderBase import RenderSkinModelBase

from TG.openGL import texture
from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

from fontTexture import GLFreetypeFace

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',

            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'Herculanum': '/Library/Fonts/Herculanum.dfont',
            'Papyrus': '/Library/Fonts/Papyrus.dfont',

            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT',

            'Helvetica': '/System/Library/Fonts/Helvetica.dfont',
            }

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True

    def renderInit(self, glCanvas, renderStart):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(8/255., 8/255., 8/255., 8/255.)
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        fontFilename = self.fonts['Helvetica']
        print fontFilename
        self.loadFontTexture(fontFilename, 24)

    def loadFontTexture(self, fontFilename, fontSize):
        font = GLFreetypeFace(fontFilename, fontSize)
        self.fontTexture = font.loadChars(string.printable)

    viewPortSize = None
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        self.viewPortSize = w, h
        glViewport (0, 0, w, h)

        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        gluOrtho2D(0, w, 0, h)

        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        if self.viewPortSize is None: return

        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        size = self.viewPortSize
        vl, vr = 0., size[0]
        vb, vt = 0., size[1]

        for e in self.fontTexture.select():
            if 0:
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            elif 0:
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
            else:
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            fw,fh = self.fontTexture.width, self.fontTexture.height

            size = self.viewPortSize
            vl = (size[0] - fw)/2.
            vr = vl + fw

            vt = size[1] - max(0, (size[1] - fh)/2.)
            vb = vt - fh

            glBegin(GL_QUADS)
            self.makeColor(vl, vr, vb, vt, c=(.5, .5, 1., 1.))
            #self.makeWhite(vl, vr, vb, vt)
            glEnd()

    def makeWhite(self, vl, vr, vb, vt):
        self.makeColor(vl, vr, vb, vt, c=(1., 1., 1., 1.))
    def makeBlack(self, vl, vr, vb, vt):
        self.makeColor(vl, vr, vb, vt, c=(0., 0., 0., 1.))

    def makeColor(self, vl, vr, vb, vt, c=(1., 1., 1., 1.)):
        glColor4f(*c);
        glNormal3s(0, 0, 1);

        glVertex3f(vl, vt, 0); glTexCoord2s(0, 1); 
        glVertex3f(vl, vb, 0); glTexCoord2s(1, 1); 
        glVertex3f(vr, vb, 0); glTexCoord2s(1, 0); 
        glVertex3f(vr, vt, 0); glTexCoord2s(0, 0);

    def makeRect(self, vl, vr, vb, vt, a=1.0):
        glNormal3s(0, 0, 1);

        glVertex3f(vl, vt, 0); glTexCoord2s(0, 1); glColor4f(0.5, 1., 1., a);
        glVertex3f(vl, vb, 0); glTexCoord2s(1, 1); glColor4f(0.5, 0.5, 1., a);
        glVertex3f(vr, vb, 0); glTexCoord2s(1, 0); glColor4f(1., 0.5, 1., a);
        glVertex3f(vr, vt, 0); glTexCoord2s(0, 0); glColor4f(1., 1., 1., a);

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


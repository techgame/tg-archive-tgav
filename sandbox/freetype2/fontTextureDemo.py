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

import array

from ctypes import byref, c_void_p

from TG.openGL import texture
from fontTexture import GLFreetypeFace
from renderBase import *

from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',
            }

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True

    def renderInit(self, glCanvas, renderStart):
        glClearColor(0.0, 0.0, 0.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

        glShadeModel(GL_SMOOTH)
        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)

        #glPolygonMode(GL_FRONT, GL_FILL)
        #glPolygonMode(GL_BACK, GL_LINE)
        #glCullFace(GL_BACK)
        #glEnable(GL_CULL_FACE)

        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if 0:
            self.quad = gluNewQuadric()
            if 1: gluQuadricDrawStyle(self.quad, GLU_FILL)
            elif 1: gluQuadricDrawStyle(self.quad, GLU_LINE)
            elif 1: gluQuadricDrawStyle(self.quad, GLU_POINT)
            gluQuadricNormals(self.quad, GLU_SMOOTH)
            gluQuadricTexture(self.quad, True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.loadCheckerBoard()

        fontFilename = self.fonts['LucidaGrande']
        fontFilename = self.fonts['Zapfino']
        print fontFilename
        self.loadFontTexture(fontFilename, 24)

        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    checkerBoard = None
    def loadCheckerBoard(self):
        return
        self.checkerBoard = texture.Texture(GL_TEXTURE_2D, GL_INTENSITY,
               wrap=GL_REPEAT, magFilter=GL_NEAREST, minFilter=GL_LINEAR)

        iSize = 1 << 8; jSize = iSize
        data = self.checkerBoard.data2d(size=(iSize, jSize), format=GL_LUMINANCE, dataType=GL_UNSIGNED_BYTE)
        data.texArray(array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                            for c in (0xff & (((i>>3) + (j>>3)) & 1) * 255,)
                                                for e in (c,)]))
        data.setImageOn(self.checkerBoard)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def loadFontTexture(self, fontFilename, fontSize):
        font = GLFreetypeFace(fontFilename, fontSize)
        self.fontTexture = font.load()
        font.loadChars('gary')

    sized = False
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()

        ho = 20./h
        wo = 20./w
        #gluOrtho2D(-1-wo, 1+wo, -1-ho, 1+ho)
        gluOrtho2D(-10, w+10, -10, h+10)
        self.viewLR = 0, w
        self.viewBT = 0, h

        #gluPerspective(60, float(w)/h, 1, 100)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()
        #gluLookAt (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0)
        #glTranslatef(0.,0.,10.)
        self.sized = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if not self.sized:
            return

        glEnable(GL_TEXTURE_2D)
        glPushMatrix()

        c = renderStart*5
        #glRotatef((c*4) % 360.0, 0, 0, 1)
        #glRotatef((c*3) % 360.0, 0, 1, 0)
        #glRotatef((c*5) % 360.0, 0, 1, 0)

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        if self.checkerBoard:
            glBindTexture(GL_TEXTURE_2D, self.checkerBoard)
        else:
            glBindTexture(GL_TEXTURE_2D, self.fontTexture)

        vl, vr = self.viewLR
        vb, vt = self.viewBT
        glBegin(GL_QUADS)

        glVertex3s(vl, vt, 0)
        glTexCoord2s(0, 1)
        glColor3f(0.5, 1, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vl, vb, 0)
        glTexCoord2s(1, 1)
        glColor3f(0.5, 0.5, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vr, vb, 0)
        glTexCoord2s(1, 0)
        glColor3f(1, 0.5, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vr, vt, 0)
        glTexCoord2s(0, 0)
        #glColor3f(1, 1, 1)
        glNormal3s(0, 0, 1)

        glEnd()

        glPopMatrix()

        glDisable(GL_TEXTURE_2D)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    RenderSkinModel().skinModel()


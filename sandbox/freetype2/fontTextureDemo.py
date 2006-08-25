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
            'StoneSans': '/Library/Fonts/Stone Sans ITC TT'
            }

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True

    def renderInit(self, glCanvas, renderStart):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

        glShadeModel(GL_SMOOTH)

        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)

        #glPolygonMode(GL_FRONT, GL_FILL)
        #glPolygonMode(GL_BACK, GL_LINE)
        #glCullFace(GL_BACK)
        #glEnable(GL_CULL_FACE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(8/255., 8/255., 8/255., 8/255.)
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

        fontFilename = self.fonts['Papyrus']
        print fontFilename
        self.loadFontTexture(fontFilename, 48)

    checkerBoard = None
    def loadCheckerBoard(self):
        self.checkerBoard = texture.Texture(GL_TEXTURE_2D, GL_INTENSITY,
               wrap=GL_REPEAT, magFilter=GL_NEAREST, minFilter=GL_LINEAR)

        iSize = 1 << 8; jSize = iSize
        data = self.checkerBoard.data2d(size=(iSize, jSize), format=GL_LUMINANCE, dataType=GL_UNSIGNED_BYTE)
        data.texArray(array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                            for c in (0xff & (((i>>3) + (j>>3)) & 1) * 128,)
                                                for e in (c,)]))
        data.setImageOn(self.checkerBoard)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def loadFontTexture(self, fontFilename, fontSize):
        font = GLFreetypeFace(fontFilename, fontSize)

        chars = string.printable
        #chars = string.lowercase+string.uppercase+string.digits
        #chars = string.lowercase+string.uppercase

        self.fontTexture = font.loadChars(chars)

    bPixelSized = True
    sized = False
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        self.viewPortSize = w, h
        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()

        if self.bPixelSized:
            gluOrtho2D(0, w, 0, h)
        else:
            if w > h:
                nw, nh = float(w)/h, 1.
            else:
                nw, nh = 1., float(h)/w
            gluOrtho2D(0, nw, 0, nh)

        #gluPerspective(60, float(w)/h, 1, 100)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()
        #gluLookAt (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0)
        #glTranslatef(0.,0.,10.)
        self.sized = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        if not self.sized: return

        c = renderStart*5

        #cc = abs(0.2 - (c/60.) % 0.4)
        #cc = 8/255
        #glClearColor(cc, cc, cc, 0)
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnable(GL_TEXTURE_2D)
        glPushMatrix()

        size = self.viewPortSize
        vl, vr = 0., size[0]
        vb, vt = 0., size[1]


        glPushMatrix()
        glTranslatef(0, 0, -.5)
        if 1:
            pass
        elif 0:
            glBindTexture(GL_TEXTURE_2D, 0)
            glBegin(GL_QUADS)
            self.makeWhite(vl, vr, vt, vb)
            glEnd()

        elif self.checkerBoard:
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glBindTexture(GL_TEXTURE_2D, self.checkerBoard)

            glBegin(GL_QUADS)
            self.makeWhite(vl, vr, vb, vt)
            glEnd()

        glPopMatrix()

        if self.fontTexture:
            if 1:
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            else:
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glBindTexture(GL_TEXTURE_2D, self.fontTexture)
            fw,fh = self.fontTexture.width, self.fontTexture.height
            if self.bPixelSized:
                size = self.viewPortSize
                vl = (size[0] - fw)/2.
                vr = vl + fw

                vt = size[1] - max(0, (size[1] - fh)/2.)
                vb = vt - fh

            else:
                if fw<fh:
                    fw,fh = 1, fh/float(fw)
                else:
                    fw,fh = fw/float(fh), 1
                vl, vr = 0, fw
                vb, vt = 0, fh

            glBegin(GL_QUADS)
            self.makeColor(vl, vr, vb, vt, c=(.5, .5, 1., 1.))
            #self.makeWhite(vl, vr, vb, vt)
            #self.makeRect(vl, vr, vb, vt)
            glEnd()
            glBegin(GL_LINE_LOOP)
            self.makeRect(vl, vr, vb, vt)
            glEnd()

            if 0:
                glTranslatef((vr+vl)/2, (vt+vb)/2, 0.5)
                glRotatef(90., 0, 0, 1)
                glTranslatef(-(vr+vl)/2, -(vt+vb)/2, 0.)

                glBegin(GL_QUADS)
                self.makeRect(vl, vr, vb, vt)
                glEnd()

        glPopMatrix()

        glDisable(GL_TEXTURE_2D)

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


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
from renderBase import *
from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
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
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        #glPolygonMode(GL_FRONT, GL_FILL)
        #glPolygonMode(GL_BACK, GL_LINE)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)

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

    def loadCheckerBoard(self):
        textureFormat, bitmapFormat = GL_LUMINANCE, GL_LUMINANCE
        itemSize = GL_UNSIGNED_BYTE
        iSize = 1 << 8; jSize = iSize
        self.checkerBoard = array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                            for c in (0xff & (i | j),)
                                                for e in (c,)])
        self.checkerBoardPtr = c_void_p(self.checkerBoard.buffer_info()[0])
        self.checkerBoardTexName = GLenum(0)
        glGenTextures(1, byref(self.checkerBoardTexName))

        texName, texPtr = (self.checkerBoardTexName, self.checkerBoardPtr)
        glBindTexture(GL_TEXTURE_2D, texName)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        if iSize <= 32:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

            glTexImage2D(GL_TEXTURE_2D, 0, textureFormat, iSize, jSize, 0, bitmapFormat, itemSize, texPtr)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

            gluBuild2DMipmaps(GL_TEXTURE_2D, textureFormat, iSize, jSize, bitmapFormat, itemSize, texPtr)

        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return
        #print 'resize:', (w, h)

        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        #gluPerspective(60, float(w)/h, 1, 100)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()
        #gluLookAt (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0)
        #glTranslatef(0.,0.,10.)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        glPushMatrix()

        #c = renderStart*5
        #glRotatef((c*4) % 360.0, 0, 0, 1)
        #glRotatef((c*3) % 360.0, 0, 1, 0)
        #glRotatef((c*5) % 360.0, 1, 0, 0)

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
        glBindTexture(GL_TEXTURE_2D, self.checkerBoardTexName)

        glBegin(GL_QUADS)

        glVertex2f(-1., 1.)
        glTexCoord2f(0., 1.)
        glColor3f(0.5, 1., 1.)
        glNormal3f(0., 0., 1.)

        glVertex2f(-1., -1.)
        glTexCoord2f(0., 0.)
        glColor3f(0.5, 0.5, 1.)
        glNormal3f(0., 0., 1.)

        glVertex2f(1., -1.)
        glTexCoord2f(1., 0.)
        glColor3f(1., 0.5, 1.)
        glNormal3f(0., 0., 1.)

        glVertex2f(1., 1.)
        glTexCoord2f(1., 1.)
        glColor3f(1., 1., 1.)
        glNormal3f(0., 0., 1.)

        glEnd()

        glPopMatrix()

        glDisable(GL_TEXTURE_2D)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    RenderSkinModel().skinModel()


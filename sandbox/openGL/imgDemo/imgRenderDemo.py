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

import os

#from ctypes import byref, c_void_p

import PIL.Image

from TG.common import path

from TG.tgavUtility.renderBase import RenderSkinModelBase
from TG.openGL import texture
from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *
from TG.openGL.raw.glext import GL_TEXTURE_RECTANGLE_ARB

imgNameList = list(path.path('.').files("*.png"))
imgMap = dict((n.basename(), n) for n in imgNameList)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    viewPortSize = None
    viewAspect = None

    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        glViewport (0, 0, w, h)
        self.viewPortSize = w, h
        self.viewPortCoords = 0, w, 0, h
        self.viewAspect = float(w) / float(h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        gluOrtho2D(0, w, 0, h)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def renderInit(self, glCanvas, renderStart):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glShadeModel(GL_SMOOTH)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if 0: # video black
            glClearColor(8/255., 8/255., 8/255., 0)
        else:
            glClearColor(1, 1, 1, 1)
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        self.imgTexure = self.loadImage(imgMap['hotbutton-selected.png'])#, GL_TEXTURE_RECTANGLE_ARB

    def renderContent(self, glCanvas, renderStart):
        if self.viewAspect is None: return

        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glPushMatrix()

        vl, vr, vt, vb = self.viewPortCoords
        for m in self.imgTexure.select():
            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
            glBegin(GL_QUADS)
            vl, vr = 0, self.imgTexure.width
            vb, vt = 0, self.imgTexure.height
            self.makeColor(vl, vr, vb, vt, c=(0., 0., 0., 0.))
            glEnd()

        glPopMatrix()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def loadImage(self, filename, texTarget=GL_TEXTURE_2D):
        img = PIL.Image.open(filename)

        nItemSize = 1
        nChannels = len(img.getbands())
        width, height = img.size

        texFormat = None
        if nChannels == 4 and img.mode == 'RGBA':
            #texFormat = GL_COMPRESSED_RGBA
            dataFormat = GL_RGBA
        elif nChannels == 3 and img.mode == 'RGB':
            #texFormat = GL_COMPRESSED_RGB
            dataFormat = GL_RGB
        else:
            raise NotImplementedError()

        imgData = img.tostring()
        assert len(imgData) == (nItemSize*nChannels*width*height)

        imgTex = texture.Texture(texTarget, texFormat or dataFormat,
               wrap=GL_CLAMP, magFilter=GL_LINEAR, minFilter=GL_LINEAR_MIPMAP_LINEAR, genMipmaps=True)
        self.glCheck()

        data = imgTex.data2d(size=img.size, format=dataFormat, dataType=GL_UNSIGNED_BYTE)
        data.texString(imgData)
        data.setImageOn(imgTex)
        self.glCheck()

        #glu.gluBuild2DMipmaps(texTarget, texFormat or dataFormat, width, height, data.format, data.dataType, None)

        return imgTex

    def makeWhite(self, vl, vr, vt, vb):
        self.makeColor(vl, vr, vt, vb, c=(1., 1., 1., 1.))
    def makeBlack(self, vl, vr, vt, vb):
        self.makeColor(vl, vr, vt, vb, c=(0., 0., 0., 1.))

    def makeColor(self, vl, vr, vb, vt, c=(1., 1., 1., 1.)):
        glColor4f(*c);
        glNormal3s(0, 0, 1);

        glVertex3f(vl, vt, 0); glTexCoord2s(0, 1);
        glVertex3f(vl, vb, 0); glTexCoord2s(1, 1);
        glVertex3f(vr, vb, 0); glTexCoord2s(1, 0);
        glVertex3f(vr, vt, 0); glTexCoord2s(0, 0);

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


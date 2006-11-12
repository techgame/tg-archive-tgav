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

from TG.common import path

from renderBase import RenderSkinModelBase
from TG.openGL.image import ImageTexture
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

    clearMask = GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT
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
        glClear(self.clearMask)

        self.img = ImageTexture(imgMap['hotbutton-selected.png'])

    def renderContent(self, glCanvas, renderStart):
        try:
            glClear(self.clearMask)
            self.img()

        except Exception:
            self.repaintTimer.Stop()
            raise

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


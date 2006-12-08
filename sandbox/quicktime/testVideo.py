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

import numpy

from renderBase import RenderSkinModelBase

from TG.openGL.image import ImageObject

from TG.openGL.raw import gl
from TG.openGL.raw.gl import *

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

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 0, w, h, -10, 10)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def renderInit(self, glCanvas, renderStart):
        glClearColor(.05, .05, .05, 1.)
        glClear(self.clearMask)

    def renderContent(self, glCanvas, renderStart):
        glClear(self.clearMask)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


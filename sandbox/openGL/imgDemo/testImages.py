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

from renderBase import RenderSkinModelBase

from TG.openGL.raw import gl
from TG.openGL.raw.gl import *

from imgObject import ImageObject

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

        glOrtho(-w/2, w/2, -h/2, h/2, -10, 10)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def renderInit(self, glCanvas, renderStart):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glClearColor(1., 1., 1., 1.)
        glClear(self.clearMask)

        spacing = 20
        totalHeight = 0
        self.imgLogo = ImageObject('tg-logo.png', align=(.5, 1.5, .5))
        totalHeight += self.imgLogo.size[1] + spacing

        self.imgButton = ImageObject('button.png', align=(.5, .5, .5))
        totalHeight += self.imgButton.size[1] + spacing

        self.imgStar = ImageObject('starshape.png', align=(.5, -.5, .5))
        totalHeight += self.imgStar.size[1] + spacing

        self.totalHeight = totalHeight - spacing
        self.spacing = spacing

    def renderContent(self, glCanvas, renderStart):
        try:
            glClear(self.clearMask)

            glLoadIdentity()

            self.imgLogo.render()
            self.imgButton.render()
            self.imgStar.render()

        except Exception:
            self.repaintTimer.Stop()
            raise

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    RenderSkinModel().skinModel()


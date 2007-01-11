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

from imgObject import ImageObject

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    viewPortSize = None
    viewAspect = None

    clearMask = gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return

        gl.glViewport (0, 0, w, h)
        self.viewPortSize = w, h
        self.viewPortCoords = 0, w, 0, h
        self.viewAspect = float(w) / float(h)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        gl.glOrtho(-w/2, w/2, -h/2, h/2, -10, 10)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def renderInit(self, glCanvas, renderStart):
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        #gl.glClearColor(1., 1., 1., 1.)
        gl.glClear(self.clearMask)

        self.imgLogo = ImageObject('tg-logo.png', align=(.5, 1.5, .5))
        self.imgButton = ImageObject('button.png', align=(.5, .5, .5))
        self.imgStar = ImageObject('starshape.png', align=(.5, -.5, .5))

    def renderContent(self, glCanvas, renderStart):
        try:
            gl.glClear(self.clearMask)

            gl.glLoadIdentity()

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


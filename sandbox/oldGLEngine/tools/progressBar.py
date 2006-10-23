##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    atype = numpy.Float16
    StartColor = numpy.array([.2, .2, .6, 0.8], atype)
    EndColor = numpy.array([.3, .7, .3, 0.8], atype)
    OutlineColor = numpy.array([1., 1., 1., 0.8], atype)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, progress=0., pos0=(-0.9, -0.05), pos1=(0.9, 0.05), scaledim=(0.99, 0.8)):
        self.pos0 = numpy.asarray(pos0, self.atype)
        self.pos1 = numpy.asarray(pos1, self.atype)
        self.scaledim = numpy.asarray(scaledim, self.atype)
        self.setProgress(progress)

    def getProgress(self, progress):
        return self.progress
    def setProgress(self, progress):
        self.progress = progress

    def glExecute(self, context):
        dim = self.pos1-self.pos0
        x0, y0 = 0.5*(self.pos1+self.pos0-self.scaledim*dim)
        dx, dy = self.scaledim*dim
        dx *= self.progress
        self._drawProgressBar((x0,y0), (x0+dx,y0+dy))
        self._drawOutline(self.pos0, self.pos1)

    def _drawProgressBar(self, (x0, y0), (x1, y1)):
        color = (1.-self.progress)*self.StartColor + self.progress*self.EndColor
        GL.glColor4fv(color)
        GL.glRectf(x0, y0, x1, y1)

    def _drawOutline(self, (x0, y0), (x1, y1)):
        GL.glColor4fv(self.OutlineColor)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glVertex2f(x0,y0)
        GL.glVertex2f(x0,y1)
        GL.glVertex2f(x1,y1)
        GL.glVertex2f(x1,y0)
        GL.glEnd()

class CenteredProgressBar(ProgressBar):
    def glExecute(self, context):
        dim = self.pos1-self.pos0
        cx, cy = 0.5*(self.pos1+self.pos0)
        dx, dy = 0.5*self.scaledim*dim
        dx *= self.progress
        self._drawProgressBar((cx-dx,cy-dy),(cx+dx,cy+dy))
        self._drawOutline(self.pos0, self.pos1)


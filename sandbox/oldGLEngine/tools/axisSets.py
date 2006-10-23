##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from OpenGL import GL

from TG.geoMath.vector import ColorVector
from TG.glEngine.attributeMgr import AttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AxisLineSet(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    attributeChange = AttributeChangeElement(GL.GL_LINE_BIT)
    lineWidth = 4.
    axisLength = 1.
    xColor = ColorVector.property((255, 0, 0, 255))
    yColor = ColorVector.property((0, 255, 0, 255))
    zColor = ColorVector.property((0, 0, 255, 255))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, axisLength=None, **kw):
        if axisLength is not None:
            self.axisLength = axisLength 
        for name, value in kw.iteritems():
            setattr(self, name, value)

    def GLExecute(self, context):
        axisLength = self.axisLength

        GL.glLineWidth(self.lineWidth)
        GL.glNormal3f(0.577,0.577,0.577)
        GL.glBegin(GL.GL_LINES)

        # X Axis
        GL.glColor4f(*self.xColor)
        GL.glVertex3f(0., 0., 0.)
        GL.glVertex3f(axisLength, 0., 0.)
        # Y Axis
        GL.glColor4f(*self.yColor)
        GL.glVertex3f(0., 0., 0.)
        GL.glVertex3f(0., axisLength, 0.)
        # Z Axis
        GL.glColor4f(*self.zColor)
        GL.glVertex3f(0., 0., 0.)
        GL.glVertex3f(0., 0., axisLength)

        GL.glEnd()
        GL.glLineWidth(1.)

        context.Statistics['lines'] = context.Statistics.get('lines', 0) + 3


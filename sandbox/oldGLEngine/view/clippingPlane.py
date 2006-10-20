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
from TG.geoMath.vector import Vector
from TG.glEngine.attributeMgr import AttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClippingPlane(object):
    attributeChange = AttributeChangeElement(GL.GL_TRANSFORM_BIT)

    planeNumber = 0
    equation = Vector.property((0.,0.,1.))

    def __init__(self, planeNumber=0, equation=None):
        self.planeNumber = planeNumber
        if equation is not None: 
            self.equation = equation

    def planeIdx(self):
        return GL.GL_CLIP_PLANE0 + self.planeNumber

    def glSelect(self, context):
        planeIdx = self.planeIdx()
        GL.glClipPlane(planeIdx, self.equation.tolist())
        context.StateMgr.Enable(planeIdx)
        
    def glDeselect(self, context):
        context.StateMgr.Disable(self.planeIdx())


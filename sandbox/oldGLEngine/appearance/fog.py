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

class FogModel(object):
    attributeChange = AttributeChangeElement(GL.GL_FOG_BIT)

    color = ColorVector.property((255,255,255,0))
    density = 1.0
    start = 0.0
    end = 1.0
    mode = GL_EXP

    def __init__(self, mode=GL_EXP, density=1.0):
        self.mode = mode
        self.density = density

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_FOG)
        GL.glFog(GL.GL_FOG_COLOR, self.color)
        GL.glFog(GL.GL_FOG_DENSITY, self.density)
        GL.glFog(GL.GL_FOG_START, self.start)
        GL.glFog(GL.GL_FOG_END, self.end)
        GL.glFog(GL.GL_FOG_MODE, self.mode)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_FOG)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FogHint(object):
    attributeChange = AttributeChangeElement(GL.GL_HINT_BIT)
    hint = GL.GL_DONT_CARE

    def glExecute(self, context):
        GL.glHint(GL.GL_FOG_HINT, self.hint)


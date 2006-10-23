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

class FaceCulling(object):
    attributeChange = AttributeChangeElement(GL.GL_POLYGON_BIT)
    frontFace = GL.GL_CCW
    cullFace = GL.GL_BACK

    def __init__(self, frontFace=None, cullFace=None):
        if frontFace is not None: self.frontFace = frontFace
        if cullFace is not None: self.cullFace = cullFace

    def glSelect(self, context):
        GL.glFrontFace(self.frontFace)
        GL.glCullFace(self.cullFace)
        context.StateMgr.Enable(GL.GL_CULL_FACE)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_CULL_FACE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolygonDrawStyle(object):
    attributeChange = AttributeChangeElement(GL.GL_POLYGON_BIT)
    frontStyle = GL.GL_FILL
    backStyle = GL.GL_FILL

    def __init__(self, frontStyle=GL.GL_FILL, backStyle=GL.GL_FILL):
        self.frontStyle = frontStyle
        self.backStyle = backStyle

    def glSelect(self, context):
        frontStyle = self.frontStyle
        backStyle = self.backStyle
        if frontStyle == backStyle:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, frontStyle)
        else:
            GL.glPolygonMode(GL.GL_FRONT, frontStyle)
            GL.glPolygonMode(GL.GL_BACK, backStyle)

    def glDeselect(self, context):
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolygonOffset(object):
    attributeChange = AttributeChangeElement(GL.GL_POLYGON_BIT)
    mode = GL.GL_POLYGON_OFFSET_FILL
    factor = 1.0
    units = 1.0

    def glSelect(self, context):
        context.StateMgr.Enable(self.mode)
        GL.glPolygonOffset(self.factor, self.units)

    def glDeselect(self, context):
        context.StateMgr.Disable(self.mode)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolygonStipple(object):
    attributeChange = AttributeChangeElement(GL.GL_POLYGON_STIPPLE_BIT)
    stipple = "\xff"*32*32 # 32x32x8 bitmask

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_POLYGON_STIPPLE)
        GL.glPolygonStipple(self.stipple)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_POLYGON_STIPPLE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolygonSmooth(object):
    attributeChange = AttributeChangeElement(GL.GL_POLYGON_BIT | GL.GL_HINT_BIT)
    smoothHint = GL.GL_DONT_CARE

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_POLYGON_SMOOTH)
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, self.smoothHint)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_POLYGON_SMOOTH)


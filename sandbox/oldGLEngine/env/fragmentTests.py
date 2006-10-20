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

from TG.geoMath.viewBox import ViewBox
from TG.glEngine.attributeMgr import AttributeChangeElement
from TG.glEngine.bufferMgr import AttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ScissorTest(object):
    attributeChange = AttributeChangeElement(GL.GL_SCISSOR_BIT)
    box = ViewBox.property()

    def glSelect(self, context):
        xywh = map(int, self.box.getXYWH())
        GL.glScissor(*xywh)
        context.StateMgr.Enable(GL.GL_SCISSOR_TEST)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_SCISSOR_TEST)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AlphaTest(object):
    attributeChange = AttributeChangeElement(GL.GL_COLOR_BUFFER_BIT)
    function = GL.GL_ALWAYS

    def __init__(self, function=GL.GL_ALWAYS):
        self.function = function

    def glSelect(self, context):
        GL.glAlphaFunc(self.function)
        context.StateMgr.Enable(GL.GL_ALPHA_TEST)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_ALPHA_TEST)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StencilTest(object):
    attributeChange = AttributeChangeElement(GL.GL_STENCIL_BUFFER_BIT)
    function = GL.GL_ALWAYS
    operation = GL.GL_KEEP

    def __init__(self, function=GL.GL_ALWAYS, operation=GL.GL_KEEP):
        self.function = function
        self.operation = operation 

    def glSelect(self, context):
        GL.glStencilFunc(self.function)
        GL.glStencilOp(self.operation)
        context.StateMgr.Enable(GL.GL_STENCIL_TEST)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_STENCIL_TEST)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DepthTest(object):
    attributeChange = AttributeChangeElement(GL.GL_DEPTH_BUFFER_BIT)
    function = GL.GL_LESS

    def __init__(self, function=GL.GL_LESS):
        self.function = function

    def glSelect(self, context):
        GL.glDepthFunc(self.function)
        context.StateMgr.Enable(GL.GL_DEPTH_TEST)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_DEPTH_TEST)


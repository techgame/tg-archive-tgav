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

class Blend(object):
    attributeChange = AttributeChangeElement(GL.GL_COLOR_BUFFER_BIT)
    src = GL.GL_SRC_ALPHA
    dst = GL.GL_ONE_MINUS_SRC_ALPHA

    def __init__(self, src=GL.GL_SRC_ALPHA, dst=GL.GL_ONE_MINUS_SRC_ALPHA):
        self.src = src
        self.dst = dst

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_BLEND)
        GL.glBlendFunc(self.src, self.dst)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_BLEND)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlendSeperateAlpha(Blend):
    srcAlpha = None
    dstAlpha = None

    class _Extensions(object):
        def __init__(self):
            from OpenGL.GL.EXT import blend_func_separate
            self.blend_func_separate = blend_func_separate.glInitBlendFuncSeparateEXT() and blend_func_separate or None
    _extensions = LazyProperty(_Extensions)

    def __init__(self, src=GL.GL_SRC_ALPHA, srcAlpha=None, dst=GL.GL_ONE_MINUS_SRC_ALPHA, dstAlpha=None):
        self.src = src
        self.srcAlpha = srcAlpha
        self.dst = dst
        self.dstAlpha = dstAlpha

    def glSelect(self, context):
        srcAlpha = self.srcAlpha 
        if srcAlpha is None: srcAlpha = self.src 
        dstAlpha = self.dstAlpha 
        if dstAlpha is None: dstAlpha = self.dst 

        context.StateMgr.Enable(GL.GL_BLEND)
        ex = self._extensions.blend_func_separate
        ex.glBlendFuncSeparateEXT(self.src, srcAlpha, self.dst, dstAlpha)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlendConstant(object):
    attributeChange = AttributeChangeElement(GL.GL_COLOR_BUFFER_BIT)
    constant = ColorVector.property((1., 1., 1., 1.))

    class _Extensions(object):
        def __init__(self):
            from OpenGL.GL.EXT import blend_color
            self.blend_color = blend_color.glInitBlendColorEXT() and blend_color or None
    _extensions = LazyProperty(_Extensions)

    def __init__(self, constant=None):
        if constant is not None:
            self.constant = constant

    def glSelect(self, context):
        ex = self._extensions.blend_color
        ex.glBlendColorEXT(*self.constant.tolist())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BlendEquation(object):
    attributeChange = AttributeChangeElement(GL.GL_COLOR_BUFFER_BIT)
    equation = None

    class _Extensions(object):
        def __init__(self):
            from OpenGL.GL.EXT import blend_minmax
            self.blend_minmax = blend_minmax.glInitBlendMinmaxEXT() and blend_minmax or None
    _extensions = LazyProperty(_Extensions)

    def __init__(self, equation):
        self.equation = equation

    def glSelect(self, context):
        ex = self._extensions.blend_minmax
        ex.glBlendEquationEXT(self.equation)


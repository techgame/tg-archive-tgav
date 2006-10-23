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
from TG.geoMath.vector import Vector, ColorVector
from TG.glEngine.attributeMgr import AttributeChangeElement
from TG.glEngine.bufferMgr import AttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClearColor(object):
    attributeChange = AttributeChangeElement(GL.GL_COLOR_BUFFER_BIT)
    bufferClear = BufferChangeElement(GL.GL_COLOR_BUFFER_BIT)
    value = ColorVector.property((0, 0, 0, 255))

    def __init__(self, Color=None):
        if Color is not None: 
            self.value = Color

    def glExecute(self, context):
        GL.glClearColor(*self.value.tolist())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorMask(object):
    mask = ColorVector.property((1., 1., 1., 1.))

    def glSelect(self, context):
        GL.glColorMask(*self.mask.tolist())

    def glDeselect(self, context):
        GL.glColorMask(1,1,1,1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClearDepth(object):
    attributeChange = AttributeChangeElement(GL.GL_DEPTH_BUFFER_BIT)
    bufferClear = BufferChangeElement(GL.GL_DEPTH_BUFFER_BIT)
    value = 1.

    def __init__(self, value=0.):
        self.value = value

    def glExecute(self, context):
        GL.glClearDepth(self.value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DepthMask(object):
    attributeChange = AttributeChangeElement(GL.GL_DEPTH_BUFFER_BIT)
    mask = 1

    def glSelect(self, context):
        GL.glDepthMask(self.mask)

    def glDeselect(self, context):
        GL.glDepthMask(1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClearStencil(object):
    attributeChange = AttributeChangeElement(GL.GL_STENCIL_BUFFER_BIT)
    bufferClear = BufferChangeElement(GL.GL_STENCIL_BUFFER_BIT)
    value = 0

    def __init__(self, value=0):
        self.value = value

    def glExecute(self, context):
        GL.glClearStencil(self.value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StencilMask(object):
    attributeChange = AttributeChangeElement(GL.GL_STENCIL_BUFFER_BIT)
    mask = -1

    def __init__(self, mask=-1):
        self.mask = mask

    def glSelect(self, context):
        GL.glStencilMask(self.mask)

    def glDeselect(self, context):
        GL.glStencilMask(-1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClearAccum(object):
    attributeChange = AttributeChangeElement(GL.GL_ACCUM_BUFFER_BIT)
    bufferClear = BufferChangeElement(GL.GL_ACCUM_BUFFER_BIT)
    value = ColorVector.property((0, 0, 0, 255))

    def __init__(self, Color=None):
        if Color is not None:
            self.value = Color

    def accumBits():
        bitflags = (GL.GL_ACCUM_RED_BITS, GL.GL_ACCUM_GREEN_BITS, GL.GL_ACCUM_BLUE_BITS, GL.GL_ACCUM_ALPHA_BITS)
        return map(GL.glGetInteger, bitflags)
    accumBits = staticmethod(accumBits)

    def glExecute(self, context):
        GL.glClearAccum(*self.value.tolist())


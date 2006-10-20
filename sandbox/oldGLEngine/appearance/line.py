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
from TG.glEngine.attributeMgr import AttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LineSize(object):
    attributeChange = AttributeChangeElement(GL.GL_LINE_BIT)
    size = 1.

    def __init__(self, size=1.):
        self.size = size

    def GLSelect(self, context):
        GL.glLineWidth(self.size)

    def GLDeselect(self, context):
        GL.glLineWidth(1.)

class LineStipple(object):
    attributeChange = AttributeChangeElement(GL.GL_LINE_BIT)
    repeat = 1
    pattern = -1

    def __init__(self, repeat=1., pattern=-1):
        self.repeat = repeat
        self.pattern = pattern

    def GLSelect(self, context):
        context.StateMgr.Enable(GL.GL_LINE_STIPPLE)
        GL.glLineStipple(self.repeat, self.pattern)

    def GLDeselect(self, context):
        context.StateMgr.Disable(GL.GL_LINE_STIPPLE)

class LineSmooth(object):
    attributeChange = AttributeChangeElement(GL.GL_LINE_BIT | GL.GL_HINT_BIT)
    smoothHint = GL.GL_DONT_CARE

    def GLSelect(self, context):
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, self.smoothHint)
        context.StateMgr.Enable(GL.GL_LINE_SMOOTH)

    def GLDeselect(self, context):
        context.StateMgr.Disable(GL.GL_LINE_SMOOTH)


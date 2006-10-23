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
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Normalization(object):
    attributeChange = AttributeChangeElement(GL.GL_TRANSFORM_BIT)

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_NORMALIZE)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_NORMALIZE)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PerspectiveHint(object):
    attributeChange = AttributeChangeElement(GL.GL_TRANSFORM_BIT)
    perspectiveHint = GL.GL_DONT_CARE

    def __init__(self, perspectiveHint=None):
        self.perspectiveHint = perspectiveHint

    def glExecute(self, context):
        GL.glHint(GL.GL_PERSPECTIVE_CORRECTION_HINT, self.perspectiveHint)


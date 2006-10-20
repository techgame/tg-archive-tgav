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

class PointSize(object):
    attributeChange = AttributeChangeElement(GL.GL_POINT_BIT)
    size = 1.

    def __init__(self, size=1.):
        self.size = size

    def glSelect(self, context):
        GL.glPointSize(self.size)

    def glDeselect(self, context):
        GL.glPointSize(1.)

class PointSmooth(object):
    attributeChange = AttributeChangeElement(GL.GL_POINT_BIT | GL.GL_HINT_BIT)
    smoothHint = GL.GL_DONT_CARE

    def glSelect(self, context):
        GL.glHint(GL.GL_POLYGON_SMOOTH_HINT, self.smoothHint)
        context.StateMgr.Enable(GL.GL_POINT_SMOOTH)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_POINT_SMOOTH)

class PointParameters(object):
    attributeChange = AttributeChangeElement(GL.GL_POINT_BIT)
    attenuation = Vector.property((1., 0., 0.))
    minSize = 0.
    maxSize = 64.
    fadeThreshold = 1.

    class _Extensions(object):
        def __init__(self):
            from OpenGL.GL.EXT import point_parameters
            self.point_parameters = point_parameters.glInitPointParametersEXT() and point_parameters or None
    _extensions = LazyProperty(_Extensions)

    def glExecute(self, context):
        pp = self._extensions.point_parameters
        pp.glPointParameterfvEXT(pp.GL_POINT_DISTANCE_ATTENUATION_EXT, self.attenuation)
        pp.glPointParameterfEXT(pp.GL_POINT_SIZE_MIN_EXT, self.minSize)
        pp.glPointParameterfEXT(pp.GL_POINT_SIZE_MAX_EXT, self.maxSize)
        pp.glPointParameterfEXT(pp.GL_POINT_FADE_THRESHOLD_SIZE_EXT, self.fadeThreshold)


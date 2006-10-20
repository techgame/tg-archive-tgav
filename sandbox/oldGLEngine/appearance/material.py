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

class Material(object):
    attributeChange = AttributeChangeElement(GL.GL_LIGHTING_BIT)

    face = GL.GL_FRONT_AND_BACK
    shininess = 0.
    Ambient = ColorVector((255,255,255,255))
    diffuse = ColorVector((255,255,255,255))
    specular = ColorVector((255,255,255,255))
    emission = ColorVector((255,255,255,255))

    def glSelect(self, context):
        face = self.face
        value = self.shininess
        if value is not None:
            GL.glMaterialf(face, GL.GL_SHININESS, value)
        value = self.ambient
        if value is not None:
            GL.glMaterialfv(face, GL.GL_AMBIENT, value)
        value = self.diffuse
        if value is not None:
            GL.glMaterialfv(face, GL.GL_DIFFUSE, value)
        value = self.specular
        if value is not None:
            GL.glMaterialfv(face, GL.GL_SPECULAR, value)
        value = self.emission
        if value is not None:
            GL.glMaterialfv(face, GL.GL_EMISSION, value)

    def glDeselect(self, context):
        pass # Ah, what to do here?

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColorMaterial(object):
    attributeChange = AttributeChangeElement(GL.GL_LIGHTING_BIT)

    parameter = GL.GL_AMBIENT_AND_DIFFUSE
    face = GL.GL_FRONT_AND_BACK

    def glSelect(self, context):
        GL.glColorMaterial(self.face, self.parameter)
        context.StateMgr.Enable(GL.GL_COLOR_MATERIAL)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_COLOR_MATERIAL)
        

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

class LightingModel(object):
    attributeChange = AttributeChangeElement(GL.GL_LIGHTING_BIT)

    localViewer = 0
    twoSided = 0
    shadeModel = GL.GL_SMOOTH
    ambientColor = ColorVector.property((64, 64, 64, 255))

    # The following is not supported at the moment by PyOpenGL -- no constants!
    #seperateSpecular = GL.GL_SINGLE_COLOR # GL.GL_SEPARATE_SPECULAR_COLOR

    def __init__(self, **kw):
        for name,value in kw.iteritems():
            setattr(self, name, value)

    def glSelect(self, context):
        context.StateMgr.Enable(GL.GL_LIGHTING)
        GL.glShadeModel(self.shadeModel)
        GL.glLightModel(GL.GL_LIGHT_MODEL_AMBIENT, self.ambientColor)
        GL.glLightModel(GL.GL_LIGHT_MODEL_TWO_SIDE, self.twoSided)
        GL.glLightModel(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, self.localViewer)

        # The following is not supported at the moment by PyOpenGL -- no constants!
        #GL.glLightModel(GL.GL_LIGHT_MODEL_COLOR_CONTROL, self.seperateSpecular)

    def glDeselect(self, context):
        context.StateMgr.Disable(GL.GL_LIGHTING)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Light(object):
    attributeChange = AttributeChangeElement(GL.GL_LIGHTING_BIT)

    lightNumber = 0

    ambient = ColorVector.property((64, 64, 64, 255))
    diffuse = ColorVector.property((255, 255, 255, 255))
    specular = ColorVector.property((255, 255, 255, 255))

    position = Vector.property((0., 0., 1., 0.))

    def __init__(self, lightNumber=0, **kw):
        self.lightNumber = lightNumber
        for name,value in kw.iteritems():
            setattr(self, name, value)

    def lightId(self):
        return GL.GL_LIGHT0 + self.lightNumber

    def glSelect(self, context):
        lightId = self.lightId()
        context.StateMgr.Enable(lightId)
        GL.glLightiv(lightId, GL.GL_AMBIENT, self.ambient)
        GL.glLightiv(lightId, GL.GL_DIFFUSE, self.diffuse)
        GL.glLightiv(lightId, GL.GL_SPECULAR, self.specular)
        GL.glLightfv(lightId, GL.GL_POSITION, self.position)

    def glDeselect(self, context):
        lightId = self.lightId()
        context.StateMgr.Disable(lightId)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AttenuatedLightMixin(object):
    attenuation = Vector.property((1., 0., 0.))

    def glSelect(self, context):
        self.__super.GLSelect(context)
        lightId = self.lightId()
        const,linear,quadratic = self.attenuation
        GL.glLightf(lightId, GL.GL_CONSTANT_ATTENUATION, const)
        GL.glLightf(lightId, GL.GL_LINEAR_ATTENUATION, linear)
        GL.glLightf(lightId, GL.GL_QUADRATIC_ATTENUATION, quadratic)

AttenuatedLightMixin._AttenuatedLightMixin__super = super(AttenuatedLightMixin)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SpotLightMixin(object):
    direction = UnitVector.property((0., 0., -1.))
    exponent = 0.0
    cutoff = 180.0

    def glSelect(self, context):
        self.__super.GLSelect(context)
        lightId = self.lightId()
        GL.glLightfv(lightId, GL.GL_SPOT_DIRECTION, self.direction)
        GL.glLightf(lightId, GL.GL_SPOT_EXPONENT, self.exponent)
        GL.glLightf(lightId, GL.GL_SPOT_CUTOFF, self.cutoff)

SpotLightMixin._SpotLightMixin__super = super(SpotLightMixin)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AttenuatedLight(AttenuatedLightMixin, Light): 
    pass
class SpotLight(SpotLightMixin, Light): 
    pass
class AttenuatedSpotLight(SpotLightMixin, AttenuatedLightMixin, Light): 
    pass


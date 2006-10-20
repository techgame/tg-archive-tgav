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

from OpenGL import GL, GLU

from TG.geoMath import xform
from TG.geoMath import projections
from TG.geoMath import quaternion
from TG.geoMath import viewBox

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
    GL.GL_COLOR_MATRIX
except AttributeError:
    GL.GL_COLOR_MATRIX = 0x80B1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if hasattr(GL, 'glLoadTransposeMatrixdARB'):
    def glExecuteAsMatrix(self, context):
        GL.glMustTransposeMatrixdARB(self.asArray4x4())
    def glExecuteLoadMatrix(self, context):
        GL.glLoadTransposeMatrixdARB(self.asArray4x4())
else:
    def glExecuteAsMatrix(self, context):
        GL.glMultMatrixd(self.asArray4x4().transpose())
    def glExecuteLoadMatrix(self, context):
        GL.glLoadMatrixd(self.asArray4x4().transpose())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ManagedMixin(object):
    _modeNameTable = {None:"< None >", GL.GL_MODELVIEW:"Model View", GL.GL_PROJECTION:"Projection", GL.GL_TEXTURE:"Texture", GL.GL_COLOR:"Color"}
    _saveMatrixLookup = {
        None: GL.GL_MODELVIEW_MATRIX,
        GL.GL_MODELVIEW: GL.GL_MODELVIEW_MATRIX,
        GL.GL_PROJECTION: GL.GL_PROJECTION_MATRIX,
        GL.GL_TEXTURE: GL.GL_TEXTURE_MATRIX,
        GL.GL_COLOR: GL.GL_COLOR_MATRIX,
        }
    mode = None
    save = False
    _save_matrix = None

    def __init__(self, mode=None, save=False, *args, **kw):
        super(ManagedMixin, self).__init__(*args, **kw)
        if mode is not None:
            self.mode = mode
        if bool(save):
            self.save = save

    def glSelect(self, context):
        if self.mode: 
            GL.glMatrixMode(self.mode)
            if self.save:
                self._saveMatrix()
            self.glExecute(context)
            GL.glMatrixMode(GL.GL_MODELVIEW)
        elif self.save:
            self._saveMatrix()
            self.glExecute(context)
        else:
            self.glExecute(context)

    def glDeselect(self, context):
        if self.save:
            if self.mode:
                GL.glMatrixMode(self.mode)
                self._restoreMatrix()
                GL.glMatrixMode(GL.GL_MODELVIEW)
            else:
                self._restoreMatrix()

    def _saveMatrix(self):
        try:
            GL.glPushMatrix()
        except GL.GLerror, e:
            self._save_matrix = GL.glGetDouble(self._saveMatrixLookup[self.mode])
        else:
            self._save_matrix = None

    def _restoreMatrix(self):
        oldmatrix = self._save_matrix
        if oldmatrix is None:
            GL.glPopMatrix()
        else:
            GL.glLoadMatrixd(oldmatrix)
            del self._save_matrix

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Composite(xform.Composite):
    def glExecute(self, context):
        for each in self.collection:
            each.glExecute(context)

class CompositeMgd(ManagedMixin, Composite):
    pass
ManagedComposite = CompositeMgd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Matrix(xform.Matrix):
    glExecute = glExecuteAsMatrix

class MatrixMgd(ManagedMixin, Matrix):
    pass

class LoadMatrix(xform.Matrix):
    glExecute = glExecuteLoadMatrix

class LoadMatrixMgd(ManagedMixin, LoadMatrix):
    pass

class Identity(xform.Identity):
    def glExecute(self, context):
        pass # Uh.... don't use this?

class IdentityMgd(ManagedMixin, Identity):
    pass

class LoadIdentity(xform.Identity):
    def glExecute(self, context):
        GL.glLoadIdentity()

class LoadIdentityMgd(ManagedMixin, LoadIdentity):
    pass

class Translate(xform.Translate):
    def glExecute(self, context):
        GL.glTranslated(*self.direction[:3])

class TranslateMgd(ManagedMixin, Translate):
    pass

class Scale(xform.Scale):
    def glExecute(self, context):
        GL.glScaled(*self.scale[:3])

class ScaleMgd(ManagedMixin, Scale):
    pass

class Rotate(xform.Rotate):
    def glExecute(self, context):
        GL.glRotated(self.angle, *self.axis[:3])

class RotateMgd(ManagedMixin, Rotate):
    pass

class Quaternion(quaternion.Quaternion):
    glExecute = glExecuteAsMatrix

class QuaternionMgd(ManagedMixin, Quaternion):
    pass

class LinearMappingMatrix(xform.LinearMappingMatrix):
    glExecute = glExecuteAsMatrix

class LinearMappingMatrixMgd(ManagedMixin, LinearMappingMatrix):
    pass

class ViewBoxMappingMatrix(viewBox.ViewBoxMappingMatrix3dh):
    glExecute = glExecuteAsMatrix

class ViewBoxMappingMatrixMgd(ManagedMixin, ViewBoxMappingMatrix):
    pass

class LookAt(xform.LookAt):
    def glExecute(self, context):
        args= self.eye.tolist() + self.center.tolist() + self.up.tolist()
        GLU.gluLookAt(*args)

class LookAtMgd(ManagedMixin, LookAt):
    pass

class SphericalLookAt(xform.SphericalLookAt):
    def glExecute(self, context):
        args= self.eye.tolist() + self.center.tolist() + self.up.tolist()
        GLU.gluLookAt(*args)

class SphericalLookAtMgd(ManagedMixin, SphericalLookAt):
    pass

class Shear(xform.Shear):
    glExecute = glExecuteAsMatrix

class ShearMgd(ManagedMixin, Shear):
    pass

class Skew(xform.Skew):
    glExecute = glExecuteAsMatrix

class SkewMgd(ManagedMixin, Skew):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Projections
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProjectionModelMixin(object):
    ProjectionModel = None

    def ApplyProjectionModel(self):
        projmodel = self.projectionModel
        if projmodel is not None: 
            projmodel.UpdateProjection(self)

class Orthographic(projections.Orthographic, ProjectionModelMixin):
    def glExecute(self, context):
        self.applyProjectionModel()
        GL.glOrtho(self.left, self.right, self.bottom, self.top, self.near, self.far)

class OrthographicMgd(ManagedMixin, Orthographic):
    pass

class Frustum(projections.Frustum, ProjectionModelMixin):
    def glExecute(self, context):
        self.applyProjectionModel()
        GL.glFrustum(self.left, self.right, self.bottom, self.top, self.near, self.far)

class FrustumMgd(ManagedMixin, Frustum):
    pass

class Perspective(projections.Perspective, ProjectionModelMixin):
    def glExecute(self, context):
        self.applyProjectionModel()
        GL.glFrustum(self.left, self.right, self.bottom, self.top, self.near, self.far)

class PerspectiveMgd(ManagedMixin, Perspective):
    pass


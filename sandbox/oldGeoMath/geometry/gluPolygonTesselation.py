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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _gluTessProperty(object):
    def __init__(self, key):
        self.key = key

    def __get__(self, obj, klass):
        if obj is None:
            return self
        else:
            tessobj = obj._getTessObj()
            return GLU.gluGetTessProperty(tessobj, self.key)

    def __set__(self, obj, value):
        tessobj = obj._getTessObj()
        GLU.gluTessProperty(tessobj, self.key, value)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class gluPolygonTesselator(object):
    """Modeled after Michael Fletcher's polygon tesselation object in OpenGLContext from pyopengl.sf.net (BSD style license)"""

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tessobj = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, *args, **settings):
        if settings:
            self.open(**settings)

        if args:
            self.tessellate(*args)

    def __del__(self):
        self.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def open(self, **settings):
        tessobj = self.tessobj = GLU.gluNewTess()
        GLU.gluTessCallback(tessobj, GLU.GLU_TESS_BEGIN, self._OnBegin)
        GLU.gluTessCallback(tessobj, GLU.GLU_TESS_VERTEX, self._OnVertex)
        GLU.gluTessCallback(tessobj, GLU.GLU_TESS_COMBINE, self._OnCombine)
        GLU.gluTessCallback(tessobj, GLU.GLU_TESS_END, self._OnEnd)

        for n, v in settings.iteritems(): 
            setattr(self, n, v)

    def close(self):
        if self.tessobj is not None:
            GLU.gluDeleteTess(self.tessobj)
        self.tessobj = None

    def tessellate(self, contours, **settings):
        tessobj = self._getTessObj()
        for n, v in settings.iteritems(): setattr(self, n, v)

        if len(contours[0][0]) == 2:
            vertexcall = lambda (x,y): GLU.gluTessVertex(tessobj, (x,y, 0), (x,y))
        else:
            vertexcall = lambda each: GLU.gluTessVertex(tessobj, each[:3], each)
        GLU.gluTessBeginPolygon(tessobj, None)
        for contour in contours:
            GLU.gluTessBeginContour(tessobj)
            map(vertexcall, contour)
            GLU.gluTessEndContour(tessobj)
        GLU.gluTessEndPolygon(tessobj)

    def GetUseEdgeFlag(self):
        try: return self._useedge
        except AttributeError: return False
    def SetUseEdgeFlag(self, value):
        if value and not callable(value):
            value = self._OnEdgeFlag
        GLU.gluTessCallback(self._getTessObj(), GLU.GLU_TESS_EDGE_FLAG, value or None)
        self._useedge = value
    useedge = property(GetUseEdgeFlag, SetUseEdgeFlag)

    def GetNormal(self):
        try: return self._normal
        except AttributeError: return (0., 0., 0.)
    def SetNormal(self, normal):
        if normal is None:
            normal = (0., 0., 0.)
        GLU.gluTessNormal(self._getTessObj(), *normal)
        self._normal = normal
    normal = property(GetNormal, SetNormal)

    windingrule = _gluTessProperty(GLU.GLU_TESS_WINDING_RULE)
    tolerance = _gluTessProperty(GLU.GLU_TESS_TOLERANCE)
    boundary = _gluTessProperty(GLU.GLU_TESS_BOUNDARY_ONLY)

    #~ protected ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _getTessObj(self):
        if self.tessobj is None:
            self.open()
        return self.tessobj

    def _OnBegin(self, mode):
        """
        if self._OnEdgeFlag is not None:
            if bool(self.boundary): 
                assert mode == GL.GL_LINE_LOOP
            else: 
                assert mode == GL.GL_TRIANGLES 
        """
        GL.glBegin(mode)

    def _OnVertex(self, vertexdata):
        GL.glVertex3fv(vertexdata)

    def _OnCombine(self, newcoord, verticies, weights):
        return newcoord

    def _OnEdgeFlag(self, edgeFlag):
        GL.glEdgeFlag(edgeFlag)

    def _OnEnd(self):
        GL.glEnd()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PolygonTesselator(gluPolygonTesselator):
    def tessellate(self, *args, **kw):
        self.results = []
        gluPolygonTesselator.tessellate(self, *args, **kw)
        return self.results

    def _OnBegin(self, mode):
        self.mode = mode
        self.vertexdata = []
    def _OnVertex(self, vertexdata):
        self.vertexdata.append(vertexdata)
    def _OnEnd(self):
        self.results.append((self.mode, self.vertexdata))
        del self.mode, self.vertexdata


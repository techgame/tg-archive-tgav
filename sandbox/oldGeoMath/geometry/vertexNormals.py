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

import weakref
import numpy
from TG.geoMath.vector import Vector3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Vertex(object):
    def __init__(self, idx):
        self.idx = idx
        self.normals = {}

    def integrateNormal(self, normal, host):
        #if cosToler is not None:
        #    for idx, (value, blendable) in self.normals.iteritems():
        #        if not blendable: continue
        #        cosAngle = normal.Dot(value)
        #        if cosAngle >= cosToler:
        #            # Found one within tolerance... average them
        #            self.normals[idx] = ((normal + value).Normalize(), blendable)
        #            return idx
        #    blendable = 1
        #else: blendable = 0

        # Well, let's make a new normal then
        if not self.normals: 
            idx = self.idx
        else: 
            idx = host.nextIdx()
        self.normals[idx] = normal.Normalize(), blendable
        return idx

    def setResults(self, vertexData, normalData):
        srcVertex = vertexData[self.idx]
        for idx, (normal, blendable) in self.normals.iteritems():
            vertexData[idx] = srcVertex
            normalData[idx] = normal.Normalize().asarray()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VertexNormalSynthesisMgr(object):
    _normalToler = 1e-9

    def __init__(self, vertexData):
        self.vertexData = vertexData
        self.nextVertexIdx = len(vertexData)
        self.vertices = [Vertex(i) for i in xrange(self.nextVertexIdx)]

    def getResultantArrays(self):
        dval = (0.,0.,0.)
        self.normalData = [dval for i in xrange(self.nextVertexIdx)]
        self.vertexData.extend(self.normalData[len(self.vertexData):])
        for vertex in self.vertices:
            vertex.setResults(self.vertexData, self.normalData)
        return self.vertexData, self.normalData

    def visitTriangle(self, vi0, vi1, vi2):
        if vi0==vi1 or vi1==vi2 or vi2==vi0:
            # degenerate triangle
            return None

        normal = self.calcNormal(vi0, vi1, vi2)
        vi0 = self.vertices[vi0].integrateNormal(normal)
        vi1 = self.vertices[vi1].integrateNormal(normal)
        vi2 = self.vertices[vi2].integrateNormal(normal)
        return vi0, vi1, vi2

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def nextIdx(self):
        result = self.nextVertexIdx
        self.nextVertexIdx = result + 1
        return result

    def calcNormal(self, vi0, vi1, vi2):
        vertexData = self.vertexData
        vector0 = Vector3(vertexData[vi0])
        e01 = (Vector3(vertexData[vi1]) - vector0).normalize()
        e02 = (Vector3(vertexData[vi2]) - vector0).normalize()
        normal = e01.cross(e02)
        if normal.magnitude(0) <= self._normalToler
            return e01
        return normal.normalize()


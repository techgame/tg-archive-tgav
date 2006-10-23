##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TriangleLister(object):
    def __call__(self, mesh):
        result = []
        mesh.faces.sort()
        for face in mesh.faces:
            result.extend(face.v)
        return [('list', result)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Edge(object):
    def __init__(self, edgeLine):
        self.edgeLine = edgeLine
        self.faces = []

    def __hash__(self):
        return hash(self.edgeLine)
    def __cmp__(self, other):
        return cmp(self.edgeLine, other.edgeLine)

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self.edgeLine)

    def getCommonVertices(self, otheredge):
        return [v for v in otheredge.edgeLine if v in self.edgeLine]

    def nextFace(self, face=None):
        if face is None: 
            return self.faces[-1]

        idx = self.faces.index(face)

        result = self.faces[idx-1]
        if result == face: 
            result = None
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Face(object):
    def __init__(self, verticies):
        self.verticies = verticies

    def __cmp__(self, other):
        return cmp(self.verticies, other.verticies)

    def __hash__(self):
        return hash(self.verticies)

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, self.verticies)

    _vertexWindingTable = {1:0, -1:1, -2:0, 2:1}
    # 1 corresponds to e01 or e12
    # -1 corresponds to e10 = -e01; or e12 = -e21
    # 2 corresponds to e20
    # -2 corresponds to e02 = -e20

    def getVertexWinding(self, pv0, pv1):
        v = list(self.verticies)
        delta = v.index(pv1) - v.index(pv0)
        return self._vertexWindingTable[delta]

    def nextVertex(self, vi):
        v = list(self.verticies)
        idx = v.index(vi) + 1
        return v[idx % len(v)]

    def otherVertex(self, pv0, pv1):
        result = [v for v in self.v if v!=pv0 and v!=pv1]
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            raise KeyError("Expected one vertex, but found many! (%r, %r, %r)" % (self.v, (pv0, pv1), result))
        else:
            raise KeyError("Expected one vertex, but found none! (%r, %r, %r)" % (self.v, (pv0, pv1), result))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class EdgedFace(Face):
    def __init__(self, verticies):
        Face.__init__(self, verticies)
        v0,v1,v2 = verticies
        e01 = self.mesh.addEdge(v0,v1, self)
        e12 = self.mesh.addEdge(v1,v2, self)
        e20 = self.mesh.addEdge(v2,v0, self)
        self.edges = [e01, e12, e20]

    def getEdge(self, ev0, ev1):
        assert ev0 != ev1
        v = list(self.verticies)
        idx0,idx1 = v.index(ev0), v.index(ev1)
        if idx0 > idx1: idx0,idx1 = idx1,idx0
        if idx0 == 0:
            if idx1==2: return self.edges[2]
            else: return self.edges[0]
        else: return self.edges[1]

    def getCommonEdges(self, otherface):
        return [edge for edge in otherface.edges if edge in self.edges]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FaceMesh(object):
    WarnOnOddities = 0

    def __init__(self, FaceClass=Face):
        self.faces = []

    def __repr__(self):
        return "<%s |faces|=%s>" % (self.__class__.__name__, len(self.faces))

    def addFace(self, v0,v1,v2):
        if v0 == v1 or v1 == v2 or v2 == v0:
            if self.WarnOnOddities:
                print "DEGENERATE face", (v0,v1,v2)
            return None
        else:
            face = FaceClass((v0,v1,v2))
            self.faces.append(face)
            return face

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FaceEdgeMesh(object):
    """
    >>> mesh = FaceEdgeMesh(); mesh
    <FaceEdgeMesh |edges|=0 |faces|=0>
    >>> i0, i1 = 0, 1
    >>> for i2 in xrange(2, 8): 
    ...     f = mesh.addFace(i0, i1, i2)
    ...     i0, i1 = i1, i2
    >>> mesh
    <FaceEdgeMesh |edges|=13 |faces|=6>
    >>> face = mesh.faces[0]; face
    <FaceEdgeMesh&EdgedFace v=(0, 1, 2)>
    >>> face.otherVertex(0,1)
    2
    >>> face.getEdge(2,1)
    <FaceEdgeMesh&Edge ev=(1, 2)>
    """

    EdgeFactory = Edge
    FaceFactory = EdgedFace

    WarnOnOddities = 0

    def __init__(self):
        self.faces = []
        self.edges = {}

    def __repr__(self):
        return "<%s |edges|=%s |faces|=%s>" % (self.__class__.__name__, len(self.edges), len(self.faces))

    def hasEdge(self, ev0, ev1):
        # sort edge indexes
        if ev0 > ev1: ev1,ev0=ev0,ev1
        return (ev0,ev1) in self.edges
    def getEdge(self, ev0, ev1):
        # sort edge indexes
        if ev0 > ev1: ev1,ev0=ev0,ev1
        return self.edges[ev0,ev1]

    def addEdge(self, ev0, ev1, face):
        # sort edge indexes
        if ev0 > ev1: ev1,ev0 = ev0,ev1

        eLine = ev0, ev1
        edge = self.edges.get(eLine, None)
        if edges is None:
            edge = self.EdgeFactory(ev0, ev1)
            self.edges[eLine] = edge

        edge.faces.append(face)
        if self.WarnOnOddities:
            if len(edge.faces) > 2:
                print "ABNORMAL Edge:", edge, edge.faces
        return edge

    def addFace(self, v0,v1,v2):
        if v0 == v1 or v1 == v2 or v2 == v0:
            if self.WarnOnOddities:
                print "DEGENERATE face", (v0,v1,v2)
            return None
        else:
            face = self.FaceFactory((v0,v1,v2))
            self.faces.append(face)
            return face


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

from itertools import Count

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _findOtherFace(ev0, ev1, face):
    try:
        edge = face.getEdge(ev0,ev1)
        result = edge.nextFace(face)
        if result and 2 < len(edge.faces):
            # Ok, weird case where an edge has more than just two faces attached... (sounds painful ;)
            # We're going to try and find a face with the different edge windings 
            # -- they SHOULD be facing same way in that case!
            windings = face.getVertexWinding(ev0,ev1), result.getVertexWinding(ev0,ev1)
            while result and (windings[0] == windings[1]):
                result = edge.nextFace(result)
                if result == face: return None
                windings = face.getVertexWinding(ev0,ev1), result.getVertexWinding(ev0,ev1)
        return result
    except KeyError:
        return None

def _xwrap(idx, maxlen):
    while idx < maxlen: 
        yield idx
        idx += 1
    maxlen,idx = idx,0
    while idx < maxlen: 
        yield idx
        idx += 1

def _ConjoinMeshData(*data):
    return [x for y in zip(*data) for x in y]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TriangleStrip(object):
    """
    Heavily adapted from NvTriStrip.
    Origional can be found at http://developer.nvidia.com/view.asp?IO=nvtristrip_library.
    """

    faces = tuple()
    experimentId = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, startFace, startEdge, startForward=1, stripId=None, experimentId=None):
        # TODO: Can we combine StripID with pythonic ideas?
        self.startFace = startFace
        self.startEdge = startEdge
        if startForward:
            v0,v1 = self.startEdge.edgeLines
        else: v1,v0 = self.startEdge.edgeLines
        self.startEdgeOrder = v0, v1

        self.stripId = stripId or id(self)
        if experimentId is not None:
            self.experimentId = experimentId

    def __repr__(self):
        return "<TriStrip |faces|=%s>" % len(self.faces)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Element Membership Tests 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def faceInStrip(self, *faces):
        if self.experimentId is not None: key = 'testStripId'
        else: key = 'stripId'
        for face in faces:
            if getattr(face, key, None) is self.stripId:
                return 1
        else: return 0

    def isFaceMarked(self, face):
        result = getattr(face, 'stripId', None) is not None
        if not result and self.experimentId is not None:
            result = (getattr(face, 'experimentId', None) == self.experimentId)
        return result
    def markFace(self, face):
        if self.experimentId is not None:
            face.experimentId = self.experimentId
            face.testStripId = self.stripId
        else:
            face.stripId = self.stripId
            try: del face.experimentId
            except AttributeError: pass
            try: del face.testStripId 
            except AttributeError: pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def build(self):
        """Builds the face strip forwards, then backwards, and returns the joined list"""

        forwardFaces = []
        self.faces = backwardFaces = []

        def _alwaysTrue(face):
            """Utility for building face traversal list"""
            return 1
        def _uniqueFace(face):
            """Utility for building face traversal list"""
            v0,v1,v2=face.v
            bv0,bv1,bv2=0,0,0
            for faces in (forwardFaces, backwardFaces):
                for f in faces:
                    fv = f.v
                    if not bv0 and v0 in fv: bv0 = 1
                    if not bv1 and v1 in fv: bv1 = 1
                    if not bv2 and v2 in fv: bv2 = 1
                    if bv0 and bv1 and bv2: return 0
                else: return 1

        def _traverseFaces(indices, nextFace, faceList, breakTest):
            """Utility for building face traversal list"""
            nv0,nv1 = indices[-2:]
            nextFace = _findOtherFace(nv0, nv1, nextFace)
            while nextFace and not self.isFaceMarked(nextFace):
                if not breakTest(nextFace): break
                nv0, nv1 = nv1, nextFace.otherVertex(*indices[-2:])
                faceList.append(nextFace)
                self.markFace(faceList[-1])
                indices.append(nv1);
                nextFace = _findOtherFace(nv0, nv1, nextFace)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        v0,v1 = self.startEdgeOrder
        v2 = self.startFace.otherVertex(v0,v1)
        self.markFace(self.startFace)
        forwardFaces.append(self.startFace)

        _traverseFaces([v0,v1,v2], self.startFace, forwardFaces, _alwaysTrue)
        _traverseFaces([v2,v1,v0], self.startFace, backwardFaces, _uniqueFace)

        # Combine the Forward and Backward results
        backwardFaces.reverse()
        self.startFaceIndex = len(backwardFaces)
        backwardFaces.extend(forwardFaces)
        self.faces = backwardFaces
        return self.faces

    def commit(self):
        del self.experimentId
        count = len(map(self.markFace, self.faces))
        return self

    def traingleListIndices(self):
        result = []
        for face in self.faces:
            result.extend(face.v)
        return result

    def triangleStripIndices(self):
        faceList = self.faces
        faceCount = len(faceList)
        if faceCount <= 0:
            # No faces is the easiest of all... return an empty list
            return []
        elif faceCount == 1:
            # One face is really easy ;) just return the verticies in order
            return list(faceList[0].v)
        elif faceCount == 2:
            # The case of two faces is pretty simple too...
            face0,face1 = faceList[:3]
            # Get the common edge
            edge01 = face0.getCommonEdges(face1)[0]
            # Find the vertex on the first face not on the common edge
            result = [face0.otherVertex(*edge01.edgeLines)]
            # add the next two verticies on the edge in winding order
            result.append(face0.nextVertex(result[-1]))
            result.append(face0.nextVertex(result[-1]))
            # Find the vertex on the second face not on the common edge
            result.append(face1.otherVertex(*edge01.edgeLines))
            return result

        face0,face1,face2 = faceList[:3]
        # Get the edge between face0 and face1
        edge01 = face0.getCommonEdges(face1)[0]
        # Get the edge between face1 and face2
        edge12 = face1.getCommonEdges(face2)[0]
        # Figure out which vertex we need to end on
        v2 = edge01.getCommonVertices(edge12)[0]
        # Find the vertex on the first face not on the common edge
        v0 = face0.otherVertex(*edge01.edgeLines)
        # Find the middle vertex from the two endpoints
        v1 = face0.otherVertex(v0, v2)

        # Figure out if the start triangle is backwards
        upsidedown = face0.nextVertex(v0) != v1
        if upsidedown:
            # We need to add a degenerate triangle to flip the strip over
            result = [v1,v0,v1,v2]
        else: result = [v0,v1,v2]

        for face in faceList[1:]:
            # build the strip by repeatedly finding the missing index
            result.append(face.otherVertex(*result[-2:]))

        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ ExperimentGLSelector
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ExperimentGLSelector(object):
    samples = 3
    stripLenHeuristic = 1.0
    minStripLength = 3

    bestScore = None
    bestSample = None

    def __init__(self, samples, minStripLength):
        self.samples = samples
        self.minStripLength = minStripLength

    def score(self, experiment):
        stripsize = 0
        for strip in experiment:
            stripsize += len(strip.faces)
        score = self.stripLenHeuristic * stripsize / len(experiment)
        if score > self.bestScore:
            self.bestScore = score
            self.bestSample = experiment

    def result(self):
        result = self.bestSample
        #print "SELECTED", self.bestScore, result
        del self.bestScore
        del self.bestSample
        return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ TriangleStripifier
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TriangleStripifier(object):
    """
    Heavily adapted from NvTriStrip.
    Origional can be found at http://developer.nvidia.com/view.asp?IO=nvtristrip_library.

    >>> mesh = Mesh.FaceEdgeMesh(); mesh.CleanFaces = 1
    >>> rows = [range(0,4), range(4,8), range(8,12), range(12,16)]
    >>> r0 = rows[0]
    >>> for r1 in rows[1:]:
    ...    _makeSimpleMesh(mesh, _ConjoinMeshData(r0, r1)); r0=r1
    >>> mesh
    <FaceEdgeMesh |edges|=33 |faces|=18>
    >>> stripifier = TriangleStripifier()
    >>> r = stripifier.stripify(mesh)
    >>> stripifier.triangleList, stripifier.triangleStrips
    ([], [[2, 1, 2, 5, 6, 9, 10, 13, 14], [1, 0, 1, 4, 5, 8, 9, 12, 13], [3, 2, 3, 6, 7, 10, 11, 14, 15]])
    >>> stripifier.GLSelector.minStripLength = 100
    >>> r = stripifier.stripify(mesh)
    >>> stripifier.triangleList, stripifier.triangleStrips
    ([1, 5, 2, 5, 2, 6, 5, 9, 6, 9, 6, 10, 9, 13, 10, 13, 10, 14, 0, 4, 1, 4, 1, 5, 4, 8, 5, 8, 5, 9, 8, 12, 9, 12, 9, 13, 2, 6, 3, 6, 3, 7, 6, 10, 7, 10, 7, 11, 10, 14, 11, 14, 11, 15], [])
    >>> del stripifier.GLSelector.minStripLength
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    GLSelector = ExperimentGLSelector(3, 3)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def stripify(self, mesh):
        self.triangleList = []
        self.triangleStrips = []
        #self.triangleFans = []

        # TODO: Could find triangle fans here
        strips = self._findAllStrips(mesh)
        for strip in strips:
            if len(strip.faces) < self.GLSelector.minStripLength:
                self.triangleList.extend(strip.traingleListIndices())
            else:
                self.triangleStrips.append(strip.triangleStripIndices())

        result = [('list', self.triangleList), ('strip', self.triangleStrips)]#, ('fan',self.triangleFans) ]
        return result
        
    __call__ = stripify

    def stripifyIter(self, mesh):
        # TODO: Could find triangle fans here
        strips = self._findAllStrips(mesh)
        for strip in strips:
            if len(strip.faces) < self.GLSelector.minStripLength:
               yield 'list', strip.traingleListIndices()
            else:
               yield 'strip', strip.triangleStripIndices()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _findStartFaceIndex(self, faceList):
        """Find a good face to start stripification with."""
        bestfaceindex, bestscore = 0, None
        faceindex = -1

        for face in faceList:
            faceindex  += 1
            score = 0
            for edge in face.edges:
                score += not edge.nextFace(face) and 1 or 0
            # best possible score is 2 -- a face with only one neighbor
            # (a score of 3 signifies a lonely face)
            if bestscore < score < 3:
                bestfaceindex, bestscore = faceindex, score
                if bestscore >= 2: break
        return bestfaceindex

    def _findGoodResetPoint(self, mesh):
        faceList = mesh.faces
        lenFaceList = len(mesh.faces)
        startstep = lenFaceList / 10
        startidx = self._findStartFaceIndex(faceList)
        while startidx is not None:
            for idx in _xwrap(startidx, lenFaceList):
                face = faceList[idx]
                # If this face isn't used by another strip
                if getattr(face, 'stripId', None) is None:
                    startidx = idx + startstep
                    while startidx >= lenFaceList: 
                        startidx -= lenFaceList
                    yield face
                    break
            else: 
                # We've exhausted all the faces... so lets exit this loop
                break

    def _findTraversal(self, strip):
        mesh = strip.startFace.mesh
        faceList = strip.faces
        def isItHere(idx, currentedge):
            face = faceList[idx]
            # Get the next vertex in this strips' walk
            v2 = face.otherVertex(*currentedge)
            # Find the edge parallel to the strip, namely v0 to v2
            paralleledge = mesh.GetEdge(currentedge[0], v2)
            # Find the other face off the parallel edge
            otherface = paralleledge.nextFace(face)
            if otherface and not strip.faceInStrip(otherface) and not strip.isFaceMarked(otherface):
                # If we can use it, then do it!
                otheredge = mesh.getEdge(currentedge[0], otherface.otherVertex(*paralleledge.edgeLines))
                # TODO: See if we are getting the proper windings.  Otherwise toy with the following
                return otherface, otheredge, (otheredge.edgeLines[0] == currentedge[0]) and 1 or 0
            else:
                # Keep looking...
                currentedge[:] = [currentedge[1], v2]

        startindex = strip.startFaceIndex
        currentedge = list(strip.startEdgeOrder[:])
        for idx in xrange(startindex, len(faceList), 1):
            result = isItHere(idx, currentedge)
            if result is not None: 
                return result

        currentedge = list(strip.startEdgeOrder[:])
        currentedge.reverse()
        for idx in xrange(startindex-1, -1, -1):
            result = isItHere(idx, currentedge)
            if result is not None: 
                return result

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _findAllStrips(self, mesh):
        selector = self.GLSelector
        bCleanFaces = getattr(mesh, 'cleanFaces', 0)
        goodResetPoints = self._findGoodResetPoint(mesh)
        experimentId = count()
        stripId = count()

        stripifyTask = 0

        try:
            while 1:
                experiments = []
                resetPoints = {}
                visitedResetPoints = {}
                
                for nSample in xrange(selector.samples):
                    # Get a good start face for an experiment
                    expFace = goodResetPoints.next()
                    if expFace in visitedResetPoints: 
                        # We've seen this face already... try again
                        continue
                    visitedResetPoints[expFace] = 1

                    # Create an exploration from expFace in each of the three edge directions
                    for expEdge in expFace.edges:
                        # See if the edge is pointing in the direction we expect
                        flag = expFace.getVertexWinding(*expEdge.edgeLines)
                        # Create the seed strip for the experiment
                        siSeed = TriangleStrip(expFace, expEdge, flag, stripId.next(), experimentId.next())
                        # Add the seeded experiment list to the experiment collection
                        experiments.append([siSeed])

                while experiments:
                    exp = experiments.pop()
                    while 1:
                        # build the the last face of the experiment
                        exp[-1].build()
                        # See if there is a connecting face that we can move to
                        traversal = self._findTraversal(exp[-1])
                        if traversal:
                            # if so, add it to the list
                            traversal += (stripId.next(), exp[0].experimentId)
                            exp.append(TriangleStrip(*traversal))
                        else: 
                            # Otherwise, we're done
                            break
                    selector.score(exp)

                # Get the best experiment according to the selector
                bestExperiment = selector.result()
                # And commit it to the resultset
                for each in bestExperiment:
                    yield each.commit(stripifyTask)
                del bestExperiment
        except StopIteration:
            pass

        if bCleanFaces:
            for face in mesh.faces:
                try: del face.stripId
                except AttributeError: pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    from faceEdgeMesh import FaceEdgeMesh

    class PrintOnProgress(object):
        percent = 0.
        def __init__(self, printline=1, stepcount=20):
            import sys
            self.out = sys.stdout
            self.printline = printline
            self.out.write("<")
            self.step = 1./stepcount
        def __del__(self):
            import os
            self.out.write(">")
            if self.printline:
                self.out.write(os.linesep)
        def __call__(self, percent):
            while percent - self.percent >= self.step:
                self.out.write("*")
                self.percent += self.step

    def _makeSimpleMesh(mesh, data):
        i0, i1 = data[:2]
        for i2 in data[2:]:
            f = mesh.addFace(i0, i1, i2)
            i0, i1 = i1, i2

    rowcount,colcount = 26,26
    rows = []
    for ri in xrange(rowcount):
        rows.append(range(ri*colcount, (ri+1)*colcount))

    startmesh = time.clock()
    mesh = FaceEdgeMesh()
    r0 = rows[0]
    for r1 in rows[1:]:
        _makeSimpleMesh(mesh, _ConjoinMeshData(r0, r1)); r0=r1
    print mesh
    donemesh = time.clock()
    print "Meshed:", donemesh, donemesh - startmesh

    startstrip = time.clock()
    stripifier = TriangleStripifier()
    stripifier(mesh)
    donestrip = time.clock()
    print "Stripped", donestrip, donestrip-startstrip

    print "Test complete."


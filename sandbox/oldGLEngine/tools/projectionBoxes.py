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

import numpy
from OpenGL import GL

from TG.geoMath.vector import ColorVector
from TG.glEngine.attributeMgr import AttributeChangeElement
from TG.glEngine.geometry.vertexArrays import VertexArray 
from TG.glEngine.geometry.arrayTraversal import IndexedCollectionTraversal  

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProjectionBox(object):
    """
    >>> ProjectionBox().update()
    array([[-1., -1., -1.],
           [ 1., -1., -1.],
           [ 1.,  1., -1.],
           [-1.,  1., -1.],
           [-1., -1.,  1.],
           [ 1., -1.,  1.],
           [ 1.,  1.,  1.],
           [-1.,  1.,  1.]],'f')
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    projection = None
    lineWidth = 4.
    attributeChange = AttributeChangeElement(GL.GL_LINE_BIT)
    _boxTraversal = IndexedCollectionTraversal(GL.GL_LINES, [[0,1, 1,2, 2,3, 3,0,   0,4, 1,5, 2,6, 3,7,   4,5, 5,6, 6,7, 7,4]], numpy.UInt8)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, projection=None):
        self.projection = projection
        self._vertices= VertexArray(numpy.zeros((8,3), numpy.Float32))

    def glExecute(self, context):
        GL.glLineWidth(self.lineWidth)
        GL.glColor4f(*self.color)
        self.update()
        self._vertices.glSelect(context)
        self._boxTraversal.glExecute(context)
        self._vertices.glDeselect(context)
        GL.glLineWidth(1.)

    def update(self):
        l,r,b,t,n,f = (-1.,1.,-1.,1.,-1.,1.)
        self._vertices.data[:4] = [(l,b,n), (r,b,n), (r,t,n), (l,t,n)]
        self._vertices.data[4:] = [(l,b,f), (r,b,f), (r,t,f), (l,t,f)]
        return self._vertices.data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class OrthographicBox(ProjectionBox):
    """
    >>> from TG.geoMath.projections import Orthographic
    >>> OrthographicBox(Orthographic(-1, 1, -1, 1, -1, 1)).update()
    array([[-1., -1., -1.],
           [ 1., -1., -1.],
           [ 1.,  1., -1.],
           [-1.,  1., -1.],
           [-1., -1.,  1.],
           [ 1., -1.,  1.],
           [ 1.,  1.,  1.],
           [-1.,  1.,  1.]],'f')
    """

    def update(self):
        l,r,b,t,n,f = [float(e) for e in self.projection.asTuple]
        self._vertices.data[:4] = [(l,b,n), (r,b,n), (r,t,n), (l,t,n)]
        self._vertices.data[4:] = [(l,b,f), (r,b,f), (r,t,f), (l,t,f)]
        return self._vertices.data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class FrustumBox(ProjectionBox):
    """
    >>> from TG.geoMath.projections import Frustum
    >>> FrustumBox(Frustum(-1, 1, -1, 1, 1, 2)).update()
    array([[-1., -1.,  1.],
           [ 1., -1.,  1.],
           [ 1.,  1.,  1.],
           [-1.,  1.,  1.],
           [-2., -2.,  2.],
           [ 2., -2.,  2.],
           [ 2.,  2.,  2.],
           [-2.,  2.,  2.]],'f')
    >>> FrustumBox(Frustum(1, 2, 1, 2, 2, 4)).update()
    array([[ 1.,  1.,  2.],
           [ 2.,  1.,  2.],
           [ 2.,  2.,  2.],
           [ 1.,  2.,  2.],
           [ 2.,  2.,  4.],
           [ 4.,  2.,  4.],
           [ 4.,  4.,  4.],
           [ 2.,  4.,  4.]],'f')
    """

    def update(self):
        l,r,b,t,n,f = [float(e) for e in self.projection.asTuple]
        self._vertices.data[:4] = [(l,b,n), (r,b,n), (r,t,n), (l,t,n)]
        scale = f/n
        l = l*scale
        r = r*scale
        b = b*scale
        t = t*scale
        self._vertices.data[4:] = [(l,b,f), (r,b,f), (r,t,f), (l,t,f)]
        return self._vertices.data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Testing 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    print "Testing..."
    import doctest
    import ProjectionBoxes as _testmod
    doctest.testmod(_testmod)
    print "Test complete."



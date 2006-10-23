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

from itertools import izip
import numpy
from OpenGL import GL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_primitveMap = {
    'pointlist': GL.GL_POINTS,

    'linelist': GL.GL_LINES,
    'lineloop': GL.GL_LINE_LOOP,
    'linestrip': GL.GL_LINE_STRIP,

    'trilist': GL.GL_TRIANGLES,
    'trifan': GL.GL_TRIANGLE_FAN,
    'tristrip': GL.GL_TRIANGLE_STRIP,

    'quadlist': GL.GL_QUADS,
    'quadstrip': GL.GL_QUAD_STRIP,
    }

_reversePrimitveMap = dict([(y,x) for x,y in _primitveMap.iteritems()])
_primitveStatsMap = {
    GL.GL_POINTS: ('points', lambda count: count),

    GL.GL_LINES: ('lines', lambda count: count//2),
    GL.GL_LINE_STRIP: ('lines', lambda count: count-1),
    GL.GL_LINE_LOOP: ('lines', lambda count: count),

    GL.GL_TRIANGLES: ('triangles', lambda count: count//3),
    GL.GL_TRIANGLE_FAN: ('triangles', lambda count: count-2), 
    GL.GL_TRIANGLE_STRIP: ('triangles', lambda count: count-2), 

    GL.GL_QUADS: ('triangles', lambda count: count//2),
    GL.GL_QUAD_STRIP: ('triangles', lambda count: count-2),
    }
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getPrimitiveCollection(primitive, count):
    if isinstance(primitive, (list, tuple)):
        assert len(primitive) == count
        return [_primitveMap.get(p, p) for p in primitive]
    else:
        return [_primitveMap.get(primitive, primitive)]*count

def addStats(self, context):
    if context.statistics:
        for statName, statResults in self._statsData.iteritems():
            context.statistics[statName] = context.statistics.get(statName,0) + statResults

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Ranged Traversals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RangedTraversalRaw(object):
    """Draws a collection of ranges using multiple primitives"""

    def __init__(self, primitives, starts, lengths):
        assert len(primitives) == len(starts) == len(lengths)
        self.primitives = primitives
        self.starts = starts
        self.lengths = lengths

        self.cacheStats()

    def cacheStats(self):
        self._statsData = {}
        for primitive, count in zip(primitives, lengths):
            statName, statCalc = _primitveStatsMap[primitive]
            self._statsData[statName] = self._statsData.get(statName, 0) + statCalc(count)
            reverseName = _reversePrimitveMap[primitive]
            self._statsData[reverseName] = self._statsData.get(reverseName, 0) + 1

    addStats = addStats

    def glExecute(self, context):
        map(GL.glDrawArrays, self.primitives, self.starts, self.lengths)
        self.addStats(context)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RangedTraversal(RangedTraversalRaw):
    """Draws a collection of ranges using multiple primitives"""

    def __init__(self, primitive, rangecollection):
        primitives = self.getPrimitiveCollection(primitive, len(rangecollection))
        starts, lengths = zip(*rangecollection)
        RangedTraversalRaw.__init__(self, primitives, starts, lengths)

    getPrimitiveCollection = staticmethod(getPrimitiveCollection)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Indexed Collection Traversals
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class IndexedCollectionTraversalRaw(object):
    """Draws a collection of index arrays using multiple primitives"""

    def __init__(self, primitives, data):
        self.primitives = primitives
        self.data = data

        self.cacheStats()

    def cacheStats(self):
        self._statsData = {}
        for primitive, datacoll in zip(primitives, data): 
            statName, statCalc = _primitveStatsMap[primitive]
            self._statsData[statName] = self._statsData.get(statName, 0) + statCalc(len(datacoll))
            reverseName = _reversePrimitveMap[primitive]
            self._statsData[reverseName] = self._statsData.get(reverseName, 0) + 1

    addStats = addStats

    def glExecute(self, context):
        map(GL.DrawElementsArray, self.primitives, self.data)
        self.addStats(context)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class IndexedCollectionTraversal(IndexedCollectionTraversalRaw):
    """Draws a collection of index arrays using multiple primitives"""

    atype = numpy.UInt16

    def __init__(self, primitive, data, atype=None):
        primitives = self.getPrimitiveCollection(primitive, len(data))
        data = [numpy.asarray(item, atype or self.atype) for item in data]
        IndexedCollectionTraversalRaw.__init__(self, primitives, data)

    getPrimitiveCollection = staticmethod(getPrimitiveCollection)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ColoredIndexedCollectionTraversal(IndexedCollectionTraversal):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    colors = []

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, primitive, data, atype=None):
        IndexedCollectionTraversal.__init__(self, primitive, data, atype)
        self.generateColors()

    def generateColors(self):
        self.colors = numpy.rand(len(self.data), 3)*0.7 + 0.3

    def glExecute(self, context):
        _glDrawElements = GL.DrawElementsArray
        for color, primitive, indexes in izip(self.colors, self.primitives, self.data):
            GL.glColor3fv(color)
            _glDrawElements(primitive, indexes)

        self.addStats(context)


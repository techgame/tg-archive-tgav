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

from itertools import count
import numpy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LocalityRemapper(object):
    def __init__(self, idxlen):
        if idxlen < 1<<7:
            format = numpy.Int8
        elif idxlen < 1<<15:
            format = numpy.Int16
        elif idxlen < 1<<31:
            format = numpy.Int32
        else: format = numpy.Int64

        self.remapIdx = numpy.array([-1], format).repeat(idxlen)
        self.remapping = numpy.zeros(idxlen, format)
        self.remappingIdx = count(0)

    def visit(self, indexreferences):
        for i in xrange(len(indexreferences)):
            idx = indexreferences[i]
            current = self.remapIdx[idx]
            if current < 0:
                current = self.remappingIdx.next()
                self.remapIdx[idx] = current
                self.remapping[current] = idx
            indexreferences[i] = current

    def remapIndices(self, datacollections):
        datacollections[:] = numpy.take(datacollections, self.remapping)


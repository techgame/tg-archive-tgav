##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy
from numpy import floor, ceil

from ..data import Rect, Vector

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BasicCell(object):
    visible = True
    box = Rect.property()

    def adjustAxisSize(self, axisSize, axis, isTrial=False):
        # axisSize must not be modified... use copies!
        return axisSize

    def layoutIn(self, pos, size):
        # pos and size must not modified... use copies!
        box = self.box
        ceil(pos, box.pos)
        floor(size, box.size)

        self.onLayout(self, box)

    def layoutHide(self):
        self.box = None

    def onLayout(self, cell, box): pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Cell(BasicCell):
    weight = Vector.property([0,0], '2f')
    minSize = Vector.property([0,0], '2f')

    def __init__(self, weight=0, min=None):
        self.weight[:] = weight

        if min is not None:
            self.minSize[:] = min

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MaxSizeCell(Cell):
    maxSize = Vector.property([0,0], '2f')

    def __init__(self, weight=0, min=None, max=None):
        Cell.__init__(self, weight, min)
        if max is not None:
            self.maxSize[:] = max

    def adjustAxisSize(self, axisSize, axis, isTrial=False):
        maxSize = self.maxSize
        idx = (maxSize > 0) & (maxSize < axisSize)
        if idx.any():
            axisSize = axisSize.copy()
            axisSize[idx] = maxSize
        return axisSize


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

    layoutVisible = False # set to true of false in layout methods
    box = Rect.property()

    def __init__(self):
        pass

    # Note: You can provide this function if you want to adjust the size
    # alloted to your cell.  If not present, some algorithms run faster
    ##def adjustAxisSize(self, axisSize, axis, isTrial=False):
    ##    # axisSize parameter must not be modified... use copies!
    ##    return axisSize

    def layoutInBox(self, lbox):
        box = self.box
        self.layoutVisible = True

        # lbox.pos and lbox.size parameters must not modified... use copies!
        ceil(lbox.pos, box.pos)
        floor(lbox.size, box.size)

        self.onlayout(self, box)

    def layoutHide(self):
        self.layoutVisible = False

    def onlayout(self, cell, box):
        pass

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    FactoryMap = {}
    @classmethod
    def register(klass, *aliases):
        for alias in aliases:
            klass.FactoryMap[alias] = klass
BasicCell.register('basic')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Cell(BasicCell):
    weight = Vector.property([0,0], '2f')
    minSize = Vector.property([0,0], '2f')

    def __init__(self, weight=None, min=None):
        BasicCell.__init__(self)
        if weight is not None:
            self.weight[:] = weight
        if min is not None:
            self.minSize[:] = min
BasicCell.register('cell')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MaxSizeCell(Cell):
    maxSize = Vector.property([0,0], '2f')

    def __init__(self, weight=None, min=None, max=None):
        Cell.__init__(self, weight, min)
        if max is not None:
            self.maxSize[:] = max

    def adjustAxisSize(self, axisSize, axis, isTrial=False):
        # axisSize parameter must not be modified... use copies!
        maxSize = self.maxSize
        idx = (maxSize > 0) & (maxSize < axisSize)
        if idx.any():
            axisSize = axisSize.copy()
            axisSize[idx] = maxSize
        return axisSize
BasicCell.register('maxsize', 'maxsizecell')


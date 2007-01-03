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

from ..data import Vector, Rect

from .cells import BasicCell
from .basicLayout import BaseLayoutStrategy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LayoutCell(BasicCell):
    layoutBox = Rect.property()

    def __init__(self, strategy='abs'):
        self.children = []
        self.strategy = strategy

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cellFactoryMap = BasicCell.FactoryMap

    def addCell(self, cell='cell', *args, **kw):
        if isinstance(cell, basestring):
            cellFactory = self.cellFactoryMap[cell]
            cell = cellFactory(*args, **kw)

        self.children.append(cell)
        return cell
    add = addCell

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __call__(self, *args, **kw):
        if args or kw:
            return self.layoutInBox(*args, **kw)
        else:
            return self.layout()

    def layout(self):
        box = self.box
        self.layoutBox = self.strategy.layout(self.children, box, False)
        self.onlayout(self, box)
        return self.layoutBox

    def layoutInBox(self, box):
        self.box.copyFrom(box)
        self.layoutVisible = True

        return self.layout()

    def layoutHide(self):
        for c in self.children:
            c.layoutHide()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    layoutStrategyMap = BaseLayoutStrategy.FactoryMap

    _strategy = None
    def getStrategy(self):
        return self._strategy
    def setStrategy(self, strategy):
        if isinstance(strategy, basestring):
            strategy = self.layoutStrategyMap[strategy]()

        self._strategy = strategy
        return strategy
    strategy = property(getStrategy, setStrategy)

LayoutCell.register('layoutcell', 'layout')
Layout = LayoutCell


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
from .basicLayout import LayoutBase

from .absLayout import AbsLayout
from .axisLayout import AxisLayout, HorizontalLayout, VerticalLayout
from .gridLayout import GridLayout, FlexGridLayout

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LayoutCell(BasicCell):
    layoutBox = Rect.property()

    def __init__(self, layoutAlg='abs'):
        self.children = []
        self.layoutAlg = layoutAlg

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    cellFactoryMap = BasicCell.FactoryMap

    def addCell(self, cell='cell'):
        if isinstance(cell, basestring):
            cell = self.cellFactoryMap[cell]()
        self.children.append(cell)
        return cell
    add = addCell

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def layoutInBox(self, box):
        self.box.copyFrom(box)
        self.layoutVisible = True

        return self.layout()

    def layoutHide(self):
        for c in self.children:
            c.layoutHide()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    layoutAlgFactoryMap = LayoutBase.FactoryMap

    _layoutAlg = None
    def getLayoutAlg(self):
        return self._layoutAlg
    def setLayoutAlg(self, layoutAlg):
        if isinstance(layoutAlg, basestring):
            layoutAlg = self.layoutAlgFactoryMap[layoutAlg]()

        self._layoutAlg = layoutAlg
        return layoutAlg
    layoutAlg = property(getLayoutAlg, setLayoutAlg)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def layout(self):
        box = self.box
        self.layoutBox = self.layoutAlg.layout(self.children, box, False)
        self.onLayout(self, box)
        return self.layoutBox

    def __call__(self, *args, **kw):
        if args or kw:
            return self.layoutInBox(*args, **kw)
        else:
            return self.layout()

LayoutCell.register('layoutcell', 'layout')


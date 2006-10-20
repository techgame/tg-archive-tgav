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

from itertools import starmap
from OpenGL import GL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SelectionEntry(object):
    __slots__ = ('_z0', '_z1', 'namepath')

    def __init__(self, z0, z1, namepath=()):
        self._z0 = z0
        self._z1 = z1
        self.namepath = namepath

    def __repr__(self):
        return "<selection z=%s names=%s>" % (self.GetZRange(), self.namepath)

    def getName(self):
        return self.getNamePath()[-1]
    def getNamePath(self):
        return self.namepath

    def getZRange(self):
        maxvalue = 0xffffffffL
        return float(self._z0)/maxvalue, float(self._z1)/maxvalue

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SelectionBuffer(object):
    EntryFactory = SelectionEntry
    size = 127

    def glSelect(self, context):
        del self.selection
        GL.glSelectBuffer(self.size)
        GL.glRenderMode(GL.GL_SELECT)
        GL.glInitNames()

    def glDeselect(self, context):
        rawselection = GL.glRenderMode(GL.GL_RENDER)
        self._setRawSelection(rawselection)

    _selection = []
    def getSelection(self):
        return self._selection
    def setSelection(self, value=()):
        self._selection = list(value)
    def delSelection(self):
        self._selection = []
    selection = property(getSelection, setSelection, delSelection)

    def _setRawSelection(self, rawSelection):
        if rawSelection:
            self.setSelection(starmap(self.EntryFactory, rawSelection))
        else:
            self.delSelection


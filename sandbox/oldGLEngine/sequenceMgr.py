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

import types
import time
import bisect
import weakref

from TG.notifications.notify import Notify
    
import attributeMgr
import bufferMgr
import stateMgr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _MatchAll(object):
    def __cmp__(self, other): return 0 # -1: less, 0: equal, 1: greater
_matchAll = _MatchAll()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SequenceBase(object):
    setupPhase = -2
    managerPhase = -1
    generalPhase = 0
    postPhase = 1

    onAddElement = Notify.objProperty()
    onRemoveElement = Notify.objProperty()
    onBeginExecute = Notify.objProperty()
    onEndExecute = Notify.objProperty()

    def __init__(self):
        self.elements = []

    def __len__(self):
        return len(self.elements)

    def addSetupElement(self, element, PriorityDelta=0):
        self._addElement(element, self.setupPhase + PriorityDelta)

    def addManagerElement(self, element, PriorityDelta=0):
        self._addElement(element, self.managerPhase + PriorityDelta)

    def addPostElement(self, element, PriorityDelta=0):
        self._addElement(element, self.postPhase + PriorityDelta)

    def addElement(self, element, priority=None):
        if priority is None: 
            priority = self.generalPhase
        self._addElement(element, priority)

    def addElements(self, elements):
        for each in elements:
            if isinstance(each, (tuple, list)):
                self.addElement(*each)
            else: self.addElement(each)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def removeElement(self, element):
        self.elements[:] = [x for x in self.elements if x[-1] != element]
        self.onRemoveElement.notify(element)

    def glExecute(self, context):
        self.onBeginExecute.notify(context)
        for priority, elementfn, element in self.elements:
            if getattr(element, 'Active', 1):
                elementfn(context)
        self.onEndExecute.notify(context)

    def _addElement(self, element, priority=0):
        if priority <= 0:
            idx = bisect.bisect_right(self.elements, (priority, _matchAll, _matchAll))
        else: 
            idx = bisect.bisect_left(self.elements, (priority, _matchAll, _matchAll))

        if isinstance(element, types.MethodType):
            elementFn, element = element, element.im_self
        elif callable(getattr(element, 'glExecute', None)):
            elementFn = element.glExecute
        elif callable(getattr(element, 'glSelect', None)):
            elementFn = element.glSelect
        else:
            raise ValueError, "Unsuppored element type %r: %r" % (type(element), element)

        self.elements.insert(idx, (priority, elementFn, element))

        try: elemAdd = element.sequenceAdd
        except AttributeError: pass
        else: elemAdd(weakref.proxy(self))

        self.onAddElement.notify(element)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Sequence(SequenceBase):
    def __init__(self, attr=0, buffer=0):
        SequenceBase.__init__(self)

        if attr:
            self.attrMgr = AttributeMgr.AttributeEffector()
        else: self.attrMgr = AttributeMgr.AttributeTracker()
        self.addManagerElement(self.attrMgr)

        if buffer:
            self.bufferMgr = BufferMgr.BufferEffector()
        else: self.bufferMgr = BufferMgr.BufferTracker()
        self.addManagerElement(self.bufferMgr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RootSequence(Sequence):
    def __init__(self):
        Sequence.__init__(self, 1, 1)
        self.stateMgr = StateMgr.StateManager()
        self.clientStateMgr = StateMgr.ClientStateManager()

    def glExecute(self, context=None):
        self.statistics = {}
        start = time.clock()
        self.statistics['start'] = start
        self.stateMgr.Reset()
        self.clientStateMgr.Reset()
        Sequence.glExecute(self, self)
        stop = time.clock()
        self.statistics['stop'] = stop
        self.statistics['duration'] = stop - start
        self.statistics['persecond'] = 1./max(1e-9, stop-start)


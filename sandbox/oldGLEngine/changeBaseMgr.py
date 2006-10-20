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

from TG.notifications.notify import Notify

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ChangeElementBase(object):
    pass

class DynamicChangeElementBase(ChangeElementBase):
    trackers = Notify.property()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ChangeTrackerBase(object):
    def __init__(self):
        self.TrackedElements = {}

    def sequenceAdd(self, Sequence):
        Sequence.OnAddElement.Add(self.OnAddElement)
        Sequence.OnRemoveElement.Add(self.OnRemoveElement)

    def onAddElement(self, Element):
        Change = getattr(Element, self._elementAttributeName, None)
        if Change is not None:
            self.TrackedElements[Change] = self.TrackedElements.get(Change, 0) + 1
            if self.TrackedElements[Change] == 1:
                Change.AddTracker(self.OnTrackedChange)

    def onRemoveElement(self, Element):
        Change = getattr(Element, self._elementAttributeName, None)
        if Change is not None:
            self.TrackedElements[Change] -= 1
            if self.TrackedElements[Change] == 0:
                del self.TrackedElements[Change]
                Change.RemoveTracker(self.OnTrackedChange)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BitmaskChangeElement(ChangeElementBase):
    """Encapsulates a single collection of bitmask changes"""

    bitmask = 0

    def __init__(self, bitmask=0):
        if bitmask:
            self.bitmask |= bitmask

    def addTracker(self, onChange):
        onChange('add', self.bitmask)

    def removeTracker(self, onChange):
        onChange('remove', self.bitmask)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BitmaskChangeTracker(ChangeTrackerBase):
    """Collects many attribute changes into a single change"""

    bitmask = 0
    default = 0
    _bitmaskNeedUpdate = True

    def __init__(self, default=None):
        ChangeTrackerBase.__init__(self)
        if default is not None:
            self.bitmask = self.default = default

    def onTrackedChange(self, ChangeType, Change):
        if ChangeType == 'add':
            self.bitmask |= Change
        elif ChangeType == 'update':
            self.bitmask |= Change # this might overestimate the needed attribute saves
            # self._bitmaskNeedUpdate = True # this might force unneeded updates.  life is so unfair ;)
        elif ChangeType == 'remove':
            self._bitmaskNeedUpdate = True
        else: 
            raise ValueError, "ChangeType is expected to be one of ['add', 'update', 'remove'], but is '%s'" % ChangeType

    def _updateBitmask(self, force=False):
        if self._bitmaskNeedUpdate or force:
            bitmask = self.default
            for each in self.TrackedELements:
                bitmask |= each.bitmask
            self.bitmask = bitmask
            self._bitmaskNeedUpdate = False

    def glExecute(self, context):
        pass
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DynamicBitmaskChangeElement(DynamicChangeElementBase):
    """Encapsulates a single collection of attribute changes"""

    _bitmask = 0

    def __init__(self, bitmask=0):
        if bitmask:
            self.bitmask |= bitmask

    def addTracker(self, onChange):
        self.trackers.Add(onChange)
        onChange('add', self.bitmask)

    def removeTracker(self, onChange):
        self.trackers.Remove(onChange)
        onChange('remove', self.bitmask)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _getBitmask(self):
        self._updateBitmask()
        return self._bitmask
    def _setBitmask(self, value):
        self._bitmask = value
        self.trackers.Update('update', self._bitmask)
    def _delBitmask(self):
        del self._bitmask
        self.trackers.Update('update', self._bitmask)
    def _updateBitmask(self, force=False):
        pass
    bitmask = property(_getBitmask, _setBitmask)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DynamicBitmaskChangeTracker(BitmaskChangeTracker, DynamicBitmaskChangeElement):
    def sequenceAdd(self, Sequence):
        result = BitmaskChangeTracker.SequenceAdd(self, Sequence)
        setattr(Sequence, self._elementAttributeName, self)
        return result


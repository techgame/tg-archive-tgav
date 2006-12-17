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

from weakref import proxy as wrproxy
from TG.observing import ObservableObject, ObservablePropertyFactoryMixin
from TG.observing.observable import ObservableProperty, AsFactoryPropAccessFctr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AsChainedFactoryPropAccessFctr(AsFactoryPropAccessFctr):
    def fdel(self):
        self.fset([])

    def fset(self, value):
        if isinstance(value, (list, tuple, basestring)):
            if self.fget().set(value):
                return

        oldValue = self.fgetattr(None)
        if oldValue is not None:
            oldValue._pub_.removeChain(self.obj, self.name)

        newValue = AsFactoryPropAccessFctr.fset(self, value)
        #self.obj._pub_.chainWith(self.name, newValue)
        newValue._pub_.addChain(self.obj, self.name)
        return newValue

class ObservableDataProperty(ObservableProperty):
    PropAccessFactoryMap = ObservableProperty.PropAccessFactoryMap.copy()
    PropAccessFactoryMap.update(
        asarraytype=AsFactoryPropAccessFctr,

        aschained=AsChainedFactoryPropAccessFctr,
        )

class ObservableData(ObservableObject, ObservablePropertyFactoryMixin):
    PropertyFactory = ObservableDataProperty
    _defaultPropKind = None

    @classmethod
    def property(klass, default=None, propKind=None, **kwbind):
        if propKind is None: 
            propKind = klass._defaultPropKind
        return klass.PropertyFactory(default, klass, propKind, **kwbind)


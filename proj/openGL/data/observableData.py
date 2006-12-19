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
from TG.observing import ObservableObject
from TG.observing.observableProperty import ObservableProperty, ObservablePropertyFactoryMixin, AsFactoryPropAccessFctr

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class AsArrayTypePropAccessFctr(AsFactoryPropAccessFctr):
    @classmethod
    def defaultAsValue(klass):
        if klass.factory is not None:
            return klass.factory(klass.default, copy=True)
        else: 
            return klass.default.copy()

class AsChainedFactoryPropAccessFctr(AsArrayTypePropAccessFctr):
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ObservableDataProperty(ObservableProperty):
    PropAccessFactoryMap = ObservableProperty.PropAccessFactoryMap.copy()
    PropAccessFactoryMap.update(
        asarraytype=AsArrayTypePropAccessFctr,
        aschainedarray=AsChainedFactoryPropAccessFctr,
        )

class ObservableData(ObservableObject, ObservablePropertyFactoryMixin):
    PropertyFactory = ObservableDataProperty
    _defaultPropKind = None

    @classmethod
    def property(klass, default=None, propKind=None, **kwbind):
        if propKind is None: 
            propKind = klass._defaultPropKind
        return klass.PropertyFactory(default, klass, propKind, **kwbind)


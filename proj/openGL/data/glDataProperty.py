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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class GLDataProperty(object):
    def __init__(self, value):
        self.value = value

    def __rawget__(self, obj):
        return getattr(obj, self.getNameIn(obj), None)
    def __rawset__(self, obj, value):
        setattr(obj, self.getNameIn(obj), value)
    def __rawdel__(self, obj):
        delattr(obj, self.getNameIn(obj))

    def __get__(self, obj, klass):
        if obj is None:
            return self

        value = self.__rawget__(obj)
        if value is None:
            value = self.__setinit__(obj)
        return value

    def __setinit__(self, obj):
        value = self.value.copy()
        self.__rawset__(obj, value)
        return value

    def __set__(self, obj, value):
        propValue = self.__get__(obj, obj.__class__)
        newValue = propValue.setPropValue(value)
        if newValue is not propValue:
            self.__rawset__(obj, newValue)

    __delete__ = __rawdel__

    _name = None
    def getNameIn(self, klass):
        name = self._name
        if name is not None:
            return name

        if not isinstance(klass, type):
            klass = klass.__class__

        for host in klass.__mro__:
            for n, v in host.__dict__.iteritems():
                if v is self:
                    self._name = '_gldp_' + n
                    return n

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def asDataProperty(host, *args, **kw):
    Property = host.PropertyFactory

    if args or kw or isinstance(host, type):
        host = host(*args, **kw)

    return Property(host)


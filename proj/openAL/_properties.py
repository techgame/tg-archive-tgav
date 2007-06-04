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

import weakref
from itertools import takewhile, count
from ctypes import cast, byref, c_void_p, _SimpleCData, POINTER

from TG.kvObserving import KVObject
from TG.openAL.raw import al, alc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multiNullString(c):
    ic = iter(c or ())
    result = (''.join(takewhile(ord, ic)) for i in count())
    result = takewhile(bool, result)
    result = list(result)
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ ALID properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALObject(object):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALIDObject(ALObject):
    _as_parameter_ = None

    def __nonzero__(self):
        return bool(self._as_parameter_)

    def _hasAsParam(self):
        return self._as_parameter_ is not None
    def _getAsParam(self):
        return self._as_parameter_
    def _setAsParam(self, asParam):
        self._as_parameter_ = asParam
    def _delAsParam(self):
        del self._as_parameter_

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALContextObject(KVObject, ALObject):
    _context = None
    def _captureCurrentContext(self):
        self._context = weakref.proxy(Context.getCurrent())

    def asWeakRef(self, cb):
        return weakref.ref(self, cb)
    def asWeakProxy(self, cb):
        return weakref.proxy(self, cb)

    def inContext(self):
        i = self._inContext(self._context)
        i.next()
        return i

    @staticmethod
    def _inContext(context):
        current = alc.alcGetCurrentContext()

        if context == current:
            yield True
        else:
            alc.alcMakeContextCurrent(context)
            yield True
            alc.alcMakeContextCurrent(current)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    processEvents = []
    _process_cache = None
    def process(self):
        cache = self._process_cache
        if cache is None:
            cache = {}
            for pe in self.processEvents:
                if not isinstance(pe, tuple):
                    pe = (pe, pe)
                cache[pe] = None
            self._process_cache = cache

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        kvpub = self.kvpub
        for (name, attr), lastVal in cache.items():
            val = getattr(self, attr or name, lastVal)
            if val != lastVal:
                kvpub(name)
            cache[name, attr] = val

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALIDContextObject(ALContextObject, ALIDObject):
    def _setAsParam(self, asParam):
        self._captureCurrentContext()
        ALIDObject._setAsParam(self, asParam)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Basic Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alBasicReadProperty(object):
    _as_parameter_ = None
    apiType = None
    apiGet = None
    byref = staticmethod(byref)

    def __init__(self, propertyEnum):
        self._as_parameter_ = propertyEnum

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiValue()
        self.apiGet(self, self.byref(apiValue))
        return self.valueFromAPI(apiValue)

    def apiValue(self, *args):
        return self.apiType(*args)

    def valueToAPI(self, pyVal):
        return pyVal

    def valueFromAPI(self, cVal):
        return cVal.value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alBasicProperty(alBasicReadProperty):
    apiSet = None

    def __set__(self, obj, value):
        apiValue = self.apiValue(self.valueToAPI(value))
        self.apiSet(self, apiValue)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Vector Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alVectorPropertyMixin(object):
    enumToCount = {}
    count = None
    apiVectorType = None # set on first use from count or enum and enumToCount
    byref = staticmethod(lambda x: x)

    def apiValue(self, *args):
        apiVectorType = self.apiVectorType
        if apiVectorType is None:
            count = self.count or self.enumToCount[self._as_parameter_]
            apiVectorType = (self.apiType * count)
            self.apiVectorType = apiVectorType
        result = apiVectorType()
        if args:
            result[:] = args[0]
        return result

    def valueToAPI(self, pyVal):
        return pyVal

    def valueFromAPI(self, cVal):
        return cVal[:]


class alVectorReadProperty(alVectorPropertyMixin, alBasicReadProperty):
    pass
class alVectorProperty(alVectorPropertyMixin, alBasicProperty):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Object Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alObjectReadProperty(alBasicReadProperty):
    def __get__(self, obj, klass):
        if obj is None: 
            return klass

        apiValue = self.apiValue()
        self.apiGet(obj, self, self.byref(apiValue))
        return self.valueFromAPI(apiValue)

class alObjectProperty(alObjectReadProperty):
    valueToAPI = lambda self, x: x
    apiSet = None

    def __set__(self, obj, value):
        apiValue = self.apiValue(self.valueToAPI(value))
        self.apiSet(obj, self, apiValue)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Vector Object Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alVectorObjectReadProperty(alVectorPropertyMixin, alObjectReadProperty):
    pass
class alVectorObjectProperty(alVectorPropertyMixin, alObjectProperty):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Applied Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alPropertyS(alBasicReadProperty):
    apiType = al.POINTER(al.ALchar)
    apiGet = staticmethod(al.alGetString)

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiGet(self)
        return self.valueFromAPI(apiValue)
    
    def valueFromAPI(self, cVal):
        return alc.cast(cVal, al.c_char_p).value

class alcPropertyI(alObjectReadProperty):
    apiType = alc.ALCint
    apiGet = staticmethod(alc.alcGetIntegerv)

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiValue()
        self.apiGet(obj, self, 1, self.byref(apiValue))
        return self.valueFromAPI(apiValue)

class alcPropertyS(alObjectReadProperty):
    apiType = alc.POINTER(alc.ALCchar)
    apiGet = staticmethod(alc.alcGetString)
    
    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiGet(obj, self)
        return self.valueFromAPI(apiValue)
    
    def valueFromAPI(self, cVal):
        return alc.cast(cVal, al.c_char_p).value

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Context import
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from context import Context


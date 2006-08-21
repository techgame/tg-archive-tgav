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
from ctypes import byref

from TG.openAL.raw import al, alc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def multiNullString(c):
    ic = iter(c or ())
    result = (''.join(takewhile(lambda ic: ord(ic.value), ic)) for i in count())
    result = takewhile(bool, result)
    result = list(result)
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ ALID properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALObject(object):
    pass

class ALIDObject(ALObject):
    __alid = None
    __alidToObj = None

    @classmethod
    def fromALID(klass, alid):
        return klass._getALIDMap().get(alid)

    @classmethod
    def _getALIDMap(klass):
        if klass.__alidToObj is None:
            klass.__alidToObj = weakref.WeakValueDictionary()
        return klass.__alidToObj

    def __nonzero__(self):
        return self._hasALID()

    def _hasALID(self):
        return self.__alid > 0

    def _getALID(self, orNone=False):
        if not self.__alid:
            if self.__alid is None:
                if orNone:
                    return None
                raise ValueError("OpenAL ID has not been created")
            elif self.__alid is False:
                if orNone:
                    return None
                raise ValueError("OpenAL ID has been destroyed")
            else:
                raise ValueError("OpenAL ID is invalid")
        return self.__alid
    def _setALID(self, alid):
        print alid, type(alid), repr(alid)
        alid = getattr(alid, 'value', alid)
        self.__alid = alid
        self._getALIDMap()[alid] = self
    def _delALID(self):
        self.__alid = False
    _alid_ = property(_getALID, _setALID, _delALID)

class ALContextObject(ALObject):
    def getObjContext(self):
        from context import Context
        return Context.fromALID(self._contextALID)

    _contextALID = None
    def _captureContextALID(self):
        self._contextALID = alc.alcGetCurrentContext()

    def withObjContext(self):
        i = self._withContext(self._contextALID)
        i.next()
        return i

    @staticmethod
    def _withContext(alidContext):
        if not isinstance(alidContext, (int, long)):
            alidContext = alidContext._alid_
        alidCurrent = alc.alcGetCurrentContext()

        if alidContext == alidCurrent:
            yield True
            yield False
        else:
            alc.alcMakeContextCurrent(alidContext)
            yield True
            alc.alcMakeContextCurrent(alidCurrent)
            yield False

class ALIDContextObject(ALContextObject, ALIDObject):
    def _setALID(self, alid):
        self._captureContextALID()
        ALIDObject._setALID(self, alid)
    _alid_ = property(ALIDObject._getALID, _setALID, ALIDObject._delALID)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Basic Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alBasicReadProperty(object):
    enum = None
    apiType = None
    apiGet = None
    byref = staticmethod(byref)

    def __init__(self, propertyEnum):
        self.enum = propertyEnum

    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiValue()
        self.apiGet(self.enum, self.byref(apiValue))
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
        self.apiSet(self.enum, apiValue)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Vector Properties
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alVectorPropertyMixin(object):
    enumToCount = {}
    count = None
    apiVectorType = None # set on first use from count or enum and enumToCount

    def apiValue(self, *args):
        apiVectorType = self.apiVectorType
        if apiVectorType is None:
            count = self.count or self.enumToCount[self.enum]
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
        self.apiGet(obj._alid_, self.enum, self.byref(apiValue))
        return self.valueFromAPI(apiValue)

class alObjectProperty(alObjectReadProperty):
    valueToAPI = lambda x: x
    apiSet = None

    def __set__(self, obj, value):
        apiValue = self.apiValue(self.valueToAPI(value))
        self.apiSet(obj._alid_, self.enum, apiValue)

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

        apiValue = self.apiGet(self.enum)
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
        self.apiGet(obj._alid_, self.enum, 1, self.byref(apiValue))
        return self.valueFromAPI(apiValue)

class alcPropertyS(alObjectReadProperty):
    apiType = alc.POINTER(alc.ALCchar)
    apiGet = staticmethod(alc.alcGetString)
    
    def __get__(self, obj, klass):
        if obj is None: 
            return self

        apiValue = self.apiGet(obj._alid_, self.enum)
        return self.valueFromAPI(apiValue)
    
    def valueFromAPI(self, cVal):
        return alc.cast(cVal, al.c_char_p).value


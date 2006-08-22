##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

__all__ = '''
    Device 
    Library 
    library 
    newContext
    '''
__all__ = __all__.split()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import weakref
import atexit

from TG.openAL._properties import *
from TG.openAL.raw import al, alc
from TG.openAL.context import Context

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Open AL Driver
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def isExtensionPresent(extName):
    return bool(al.alIsExtensionPresent(str(extName)))

def enumValue(enumName):
    return int(al.alGetEnumValue(str(enumName)))

class Library(object):
    version = alPropertyS(al.AL_VERSION)
    vendor = alPropertyS(al.AL_VENDOR)
    renderer = alPropertyS(al.AL_RENDERER)

    def getExtensions(self):
        return set(self.rawExtensions.split(' '))
    extensions = property(getExtensions)
    rawExtensions = alPropertyS(al.AL_EXTENSIONS)

    isExtensionPresent = staticmethod(isExtensionPresent)
    enumValue = staticmethod(enumValue)
    
    def destory(self):
        self.delSoftwareDC()
        self.delHardwareDC()

    def newContext(self, deviceName=None, attrs=[], makeCurrent=True):
        return Device(deviceName).newContext(attrs, makeCurrent=makeCurrent)

    _softwareDC = None
    def getSoftwareDC(self):
        if self._softwareDC is None:
            self.setSoftwareDC(self.newContext('Generic Software'))
        return self._softwareDC
    def setSoftwareDC(self, dc):
        self._softwareDC = dc
    def delSoftwareDC(self):
        self._softwareDC = None
    softwareDC = property(getSoftwareDC)

    _hardwareDC = None
    def getHardwareDC(self):
        if self._hardwareDC is None:
            self.setHardwareDC(self.newContext('Generic Hardware'))
        return self._hardwareDC
    def setHardwareDC(self, dc):
        self._hardwareDC = dc
    def delHardwareDC(self):
        self._hardwareDC = None
    hardwareDC = property(getHardwareDC, setHardwareDC, delHardwareDC)

library = Library()
atexit.register(library.destory)

newContext = library.newContext

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ AL Device Drivers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Device(ALIDObject):
    ContextFactory = Context

    name = alcPropertyS(alc.ALC_DEVICE_SPECIFIER)

    versionMajor = alcPropertyI(alc.ALC_MAJOR_VERSION)
    versionMinor = alcPropertyI(alc.ALC_MINOR_VERSION)

    def __init__(self, name=None):
        if name is not False:
            self.open(name)

    def __del__(self):
        self.close()

    def open(self, name=None):
        name = name and str(name) or None
        self._alid_ = alc.alcOpenDevice(name)
        return self._hasALID()

    __dealocating = False
    def close(self):
        if self._hasALID() and not self.__dealocating:
            self.__dealocating = True
            try:
                self._delContexts()
                alc.alcCloseDevice(self._alid_)
            finally:
                del self._alid_
    destroy = close

    @classmethod
    def fromCurrentContext(klass):
        return klass.fromALID(alc.alcGetCurrentContext())

    @classmethod
    def fromContext(klass, context):
        return klass.fromALID(context._alid_)

    def newContext(self, attrs=[], makeCurrent=True):
        context = self.ContextFactory(self, attrs)
        if makeCurrent:
            context.makeCurrent()
        return context

    _contextMap = None
    def _getContextMap(self):
        if self._contextMap is None:
            self._setContextMap({}) #weakref.WeakValueDictionary())
        return self._contextMap
    def _setContextMap(self, contextCollection):
        self._contextMap = contextCollection
    def _delContexts(self):
        if self._contextMap is not None:
            for context in self._contextMap.itervalues():
                context.destroy()
            del self._contextMap

    def _addContext(self, context):
        self._getContextMap()[context._alid_] = context
    def _removeContext(self, context):
        self._getContextMap().pop(context._alid_)
    
    def getVersion(self):
        return '%s.%s' % (self.versionMajor, self.versionMinor)
    version = property(getVersion)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def defaultDeviceName(klass):
        cVal = alc.alcGetString(None, alc.ALC_DEFAULT_DEVICE_SPECIFIER)
        return alc.cast(cVal, alc.c_char_p).value
    
    @classmethod
    def defaultDevice(klass):
        return klass()

    @classmethod
    def allDeviceNames(klass):
        cVal = alc.alcGetString(None, alc.ALC_DEVICE_SPECIFIER)
        return multiNullString(cVal)

    @classmethod
    def allDevices(klass):
        devices = [klass(name) for name in klass.allDeviceNames()]
        if not devices:
            devices = [klass()]
        return devices

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


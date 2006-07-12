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

from TG.openAL._properties import *
from TG.openAL.raw import al, alc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Context(ALIDObject):
    def __init__(self, device=None, attrs=[]):
        if device is not False:
            self.create(device, attrs)
        
    def __del__(self):
        try:
            self.destroy()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def create(self, device=None, attrs=[]):
        """Attrs is a packed list of:
            ALC_FREQUENCY:
                Frequency for mixing output buffer, in units of Hz 
            ALC_REFRESH:
                Refresh intervals, in units of Hz 
            ALC_SYNC:
                Flag, indicating a synchronous context 
            ALC_MONO_SOURCES:
                A hint indicating how many sources should be capable of supporting mono data 
            ALC_STEREO_SOURCES:
                A hint indicating how many sources should be capable of supporting stereo data 
        """
        if self._hasALID():
            raise Exception("Context already initialized")

        if attrs:
            attrs = map(int, attrs) + [0]
        else: attrs = None

        self._alid_ = alc.alcCreateContext(device._alid_, attrs)
        device._addContext(self)
        self._device = device

    __dealocating = False
    def destroy(self):
        if self._hasALID() and not self.__dealocating:
            self.__dealocating = True
            try:
                self.makeCurrent()
                self.delSources()
                self.delBuffers()

                alc.alcDestroyContext(self._alid_)

            finally:
                del self._alid_
                self._device = None

    def makeCurrent(self):
        return self._makeCurrentALID(self._alid_)

    @classmethod
    def _makeCurrentALID(klass, alid):
        return bool(alc.alcMakeContextCurrent(alid))

    def process(self):
        alc.alcProcessContext(self._alid_)
    def suspend(self):
        alc.alcSuspendContext(self._alid_)

    @classmethod
    def getCurrent(klass):
        return klass.fromALID(alc.alcGetCurrentContext())
    def getDevice(self):
        result = self._device
        if result is None:
            result = Device.fromALID(alc.alcGetContextsDevice(self._alid_))
        return result

    def getAttrs(self):
        return self._getAttrsFor(self._alid_)

    @classmethod
    def getDefaultAttrs(self):
        return self._getAttrsFor(None)

    @classmethod
    def _getAttrsFor(klass, alid):
        bytes = (alc.ALCint*1)()
        alc.alcGetIntegerv(alid, alc.byref(bytes))
        bytes = (alc.ALCint*bytes[0])()
        alc.alcGetIntegerv(alid, alc.byref(bytes))
        return bytes[:]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Source Collection
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _sources = None
    def getSources(self):
        if self._sources is None:
            self.setSources(set())
        return self._sources
    def setSources(self, sources):
        self._sources = sources
    def delSources(self):
        if self._sources is not None:
            while self._sources:
                src = self._sources.pop()
                src.destroy()
            del self._sources

    def addSource(self, source):
        self.getSources().add(source)
    def removeSource(self, source):
        self.getSources().discard(source)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Buffer Collection
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _buffers = None
    def getBuffers(self):
        if self._buffers is None:
            self.setBuffers(set())
        return self._buffers
    def setBuffers(self, buffers):
        self._buffers = buffers
    def delBuffers(self):
        if self._buffers is not None:
            while self._buffers:
                buf = self._buffers.pop()
                buf.destroy()
            del self._buffers

    def addBuffer(self, buffer):
        self.getBuffers().add(buffer)
    def removeBuffer(self, buffer):
        self.getBuffers().discard(buffer)


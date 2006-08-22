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
from TG.openAL.raw import al, alc, alut

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

alFormatFromChannels = {
    (1,1): al.AL_FORMAT_MONO8,
    (1,8): al.AL_FORMAT_MONO8,

    (1,2): al.AL_FORMAT_MONO16,
    (1,16): al.AL_FORMAT_MONO16,

    (2,1): al.AL_FORMAT_STEREO8,
    (2,8): al.AL_FORMAT_STEREO8,

    (2,2): al.AL_FORMAT_STEREO16,
    (2,16): al.AL_FORMAT_STEREO16,
    }

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _alBufferPropertyI(alObjectProperty):
    apiType = al.ALint
    apiGet = staticmethod(al.alGetBufferi)
    if hasattr(al, 'alBufferi'):
        apiSet = staticmethod(al.alBufferi)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Buffer(ALIDContextObject):
    frequency = _alBufferPropertyI(al.AL_FREQUENCY)
    bits = _alBufferPropertyI(al.AL_BITS)
    channels = _alBufferPropertyI(al.AL_CHANNELS)
    size = _alBufferPropertyI(al.AL_SIZE)

    def __init__(self, bCreate=True):
        if not bCreate:
            return
        self.create()

    def __repr__(self):
        result = "<%s.%s alid: %s" % (
                self.__class__.__module__,
                self.__class__.__name__,
                self._getALID(True))
        if self._hasALID():
            result += " freq: %s bits: %s channels: %s size: %s>" % (
                    self.frequency, self.bits, self.channels, self.size)
        else: 
            result += ">"
        return result

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def create(self):
        if self._hasALID():
            raise Exception("Buffer has already been created")

        bufferIds = (1*al.ALuint)()
        al.alGenBuffers(1, bufferIds)
        self.createFromId(bufferIds[0])
        return self

    def createFromId(self, bufferId):
        self._alid_ = bufferId
        self.getObjContext().addBuffer(self)
        return self

    def destroy(self):
        if self._hasALID():
            i = self.withObjContext()
            try:
                self.destroyFromId(self._alid_)
            finally:
                del self._alid_
                i.next()

    def destroyFromId(self, bufferId):
        bufferIds = (1*al.ALuint)()
        bufferIds[:] = [bufferId]
        al.alDeleteBuffers(1, bufferIds)

        ctx = self.getObjContext()
        if ctx is not None:
            ctx.removeBuffer(self)

    def setData(self, data, format, frequency):
        return self.setDataRaw(data, len(data), format, frequency)

    def setDataFromChannels(self, data, channels, width, frequency):
        format = alFormatFromChannels[channels, width]
        return self.setData(data, format, frequency)

    def setDataFromWave(self, waveReader):
        return self.setDataFromChannels(waveReader.readFrames(), waveReader.channels, waveReader.width, waveReader.frequency)

    def setDataFromCapture(self, data, capture):
        return self.setData(data, capture.format, capture.frequency)

    def setDataRaw(self, data, size, format, frequency):
        format = format or self.format
        frequency = frequency or self.frequency
        al.alBufferData(self._alid_, format, data, size, frequency)
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _queuedSrcs = None
    def getQueuedSources(self):
        if self._queuedSrcs is None:
            self._queuedSrcs = set()
        return self._queuedSrcs

    def onQueued(self, src):
        self.getQueuedSources().add(src)
    def onDequeued(self, src):
        self.getQueuedSources().discard(src)

    def queue(self, *srcs):
        for src in srcs:
            src.queue(self)
    def dequeue(self, *srcs):
        srcs = srcs or self.getQueuedSources().copy()
        for src in srcs:
            try:
                src.dequeue(self)
                continue
            except al.ALException, e:
                if e.error != al.AL_INVALID_VALUE:
                    raise

            src.stop()
            src.dequeue(self)

    def play(self):
        from TG.openAL.source import Source
        self.playOn(Source(self))

    def playOn(self, *srcs):
        for src in srcs:
            src.playQueue(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromFilename(klass, filename):
        self = klass()
        self.loadFilename(filename)
        return self

    def loadFilename(self, filename):
        """Make sure that the source is dequeued from all buffers"""
        format = al.ALenum(0)
        data = al.POINTER(al.ALvoid)()
        size = al.ALsizei(0)
        freqency = al.ALsizei(0)
        loop = al.ALboolean(0)

        alut.alutLoadWAVFile(str(filename), al.byref(format), al.byref(data), al.byref(size), al.byref(freqency))#, al.byref(loop))
        try:
            self.loadPCMData(format, data, size, freqency)
        finally:
            alut.alutUnloadWAV(format, data, size, freqency)
        return self

    @classmethod
    def fromFile(klass, dataFile):
        self = klass()
        return self.loadFile(dataFile)

    def loadFile(self, dataFile):
        dataFile.seek(0)
        raw = dataFile.read()
        return self.loadData(raw)

    @classmethod
    def fromData(klass, data):
        self = klass()
        self.loadData(data)
        return self

    def loadWaveData(self, waveRaw):
        format = al.ALenum(0)
        data = al.POINTER(al.ALvoid)()
        size = al.ALsizei(0)
        freqency = al.ALsizei(0)
        loop = al.ALboolean('\x00')

        alut.alutLoadWAVMemory(waveRaw, al.byref(format), al.byref(data), al.byref(size), al.byref(freqency))#, al.byref(loop))
        try:
            self.loadPCMData(format, data, size, freqency)
        finally:
            alut.alutUnloadWAV(format, data, size, freqency)
        return self
    loadData = loadWaveData

    def loadPCMData(self, format, pcmData, size, freqency):
        #pcmData = al.POINTER(al.ALvoid)()
        if not isinstance(format, al.ALenum):
            format = al.ALenum(format)
        if not isinstance(size, al.ALsizei):
            size = al.ALsizei(size)
        if not isinstance(freqency, al.ALsizei):
            freqency = al.ALsizei(freqency)

        self.dequeue()
        al.alBufferData(self._alid_, format, pcmData, size, freqency)
        return self

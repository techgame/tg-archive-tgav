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
from TG.openAL.constants import alFormatMap
from TG.openAL.raw import al, alc, alut
from TG.openAL.raw.errors import ALException

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
    processEvents = [
        'playingSources',
        'queuedSources',

        'format',
        'frequency',
        'channels',
        'size',
        ]

    frequency = _alBufferPropertyI(al.AL_FREQUENCY)
    bits = _alBufferPropertyI(al.AL_BITS)
    channels = _alBufferPropertyI(al.AL_CHANNELS)
    size = _alBufferPropertyI(al.AL_SIZE)

    def getSamples(self):
        return self.size * 8 / (self.bits*self.channels)
    samples = property(getSamples)

    def getSeconds(self):
        return self.getSamples()/float(self.frequency)
    seconds = property(getSeconds)

    _allocated = True

    def __init__(self, bCreate=True):
        if not bCreate:
            return
        self.create()

    def __repr__(self):
        result = "<%s.%s alid: %s" % (
                self.__class__.__module__,
                self.__class__.__name__,
                self._as_parameter_)
        if self._hasAsParam():
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
        if self._hasAsParam():
            raise Exception("Buffer has already been created")

        bufferIds = (1*al.ALuint)()
        al.alGenBuffers(1, bufferIds)
        self.createFromId(bufferIds[0])
        return self

    @classmethod
    def fromId(klass, bufferId):
        if bufferId:
            self = klass(False)
            self._setAsParam(bufferId)
            self._allocated = False
            return self
        else: return None

    def createFromId(self, bufferId):
        self._setAsParam(bufferId)
        #self._context.addBuffer(self)
        return self

    def destroy(self):
        if not self._allocated:
            return 

        if self._hasAsParam():
            i = self.inContext()
            try:
                self.destroyFromId(self)
            finally:
                self._delAsParam()
                del i

    def destroyFromId(self, bufferId):
        bufferIds = (1*al.ALuint)()
        bufferIds[:] = [bufferId._as_parameter_]
        bufferId._as_parameter_ = None
        try:
            al.alDeleteBuffers(1, bufferIds)
        except ALException:
            pass

        #ctx = self._context
        #if ctx is not None:
        #    ctx.removeBuffer(self)

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
        format = alFormatMap.get(format, format)
        frequency = frequency or self.frequency
        al.alBufferData(self, format, data, size, frequency)
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _playingSrcs = None
    def getPlayingSources(self):
        if self._playingSrcs is None:
            self._playingSrcs = set()
        return self._playingSrcs
    playingSources = property(getPlayingSources)

    _queuedSrcs = None
    def getQueuedSources(self):
        if self._queuedSrcs is None:
            self._queuedSrcs = set()
        return self._queuedSrcs
    queuedSources = property(getQueuedSources)

    def isQueued(self):
        return bool(self._queuedSrcs)

    def onQueued(self, src):
        self.getQueuedSources().add(src)
        self.process()
    def onDequeued(self, src):
        self.getQueuedSources().discard(src)
        self.process()

    def queue(self, *srcs):
        for src in srcs:
            src.queue(self)
    def dequeue(self, *srcs):
        srcs = srcs or self.getQueuedSources().copy()
        for src in srcs:
            try:
                src.dequeue(self)
                continue
            except ALException, e:
                if e.error != al.AL_INVALID_VALUE:
                    raise

            src.stop()
            src.dequeue(self)

    def play(self):
        from TG.openAL.source import Source
        return self.playOn(Source(self))

    def playOn(self, *srcs):
        for src in srcs:
            src.playQueue(self)
        return src

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def fromFilename(klass, filename):
        self = klass()
        self.loadFilename(filename)
        return self

    def loadFilename(self, filename):
        """Make sure that the buffer is dequeued from all sources"""
        format = al.ALenum(0)
        data = al.POINTER(al.ALvoid)()
        size = al.ALsizei(0)
        frequency = al.ALsizei(0)

        alut.alutLoadWAVFile(str(filename), al.byref(format), al.byref(data), al.byref(size), al.byref(frequency))
        try:
            self.loadPCMData(format, data, size, frequency)
        finally:
            alut.alutUnloadWAV(format, data, size, frequency)
        return self

    @classmethod
    def fromFile(klass, dataFile):
        self = klass()
        return self.loadFile(dataFile)

    def loadFile(self, dataFile):
        """Make sure that the buffer is dequeued from all sources"""
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
        frequency = al.ALsizei(0)

        alut.alutLoadWAVMemory(waveRaw, al.byref(format), al.byref(data), al.byref(size), al.byref(frequency))
        try:
            self.loadPCMData(format, data, size, frequency)
        finally:
            alut.alutUnloadWAV(format, data, size, frequency)
        return self
    loadData = loadWaveData

    def loadPCMData(self, format, pcmData, size, frequency):
        if not isinstance(format, al.ALenum):
            format = alFormatMap.get(format, format)
            format = al.ALenum(format)
        if not isinstance(size, al.ALsizei):
            size = al.ALsizei(size)
        if not isinstance(frequency, al.ALsizei):
            frequency = al.ALsizei(frequency)

        self.dequeue()
        al.alBufferData(self, format, pcmData, size, frequency)
        return self


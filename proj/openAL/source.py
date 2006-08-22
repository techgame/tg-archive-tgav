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
from TG.openAL.raw import al

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class alSourcePropertyI(alObjectProperty):
    apiType = al.ALint
    apiGet = staticmethod(al.alGetSourcei)
    apiSet = staticmethod(al.alSourcei)

class alSourcePropertyF(alObjectProperty):
    apiType = al.ALfloat
    apiGet = staticmethod(al.alGetSourcef)
    apiSet = staticmethod(al.alSourcef)

class alSourcePropertyFV(alVectorObjectProperty):
    enumToCount = {
        al.AL_POSITION: 3,
        al.AL_VELOCITY: 3,
        al.AL_DIRECTION: 3,
    }
    apiType = al.ALfloat
    apiGet = staticmethod(al.alGetSourcefv)
    apiSet = staticmethod(al.alSourcefv)

    def byref(self, item):
        return item

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Source(ALIDContextObject):
    position = alSourcePropertyFV(al.AL_POSITION)
    velocity = alSourcePropertyFV(al.AL_VELOCITY)
    direction = alSourcePropertyFV(al.AL_DIRECTION)

    relative = alSourcePropertyF(al.AL_SOURCE_RELATIVE)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pitch = alSourcePropertyF(al.AL_PITCH)

    gain = alSourcePropertyF(al.AL_GAIN)
    min_gain = alSourcePropertyF(al.AL_MIN_GAIN)
    max_gain = alSourcePropertyF(al.AL_MAX_GAIN)
    cone_outer_gain = alSourcePropertyF(al.AL_CONE_OUTER_GAIN)
    cone_outer_angle = alSourcePropertyF(al.AL_CONE_OUTER_ANGLE)
    cone_inner_angle = alSourcePropertyF(al.AL_CONE_INNER_ANGLE)

    max_distance = alSourcePropertyF(al.AL_MAX_DISTANCE)
    rolloff_factor = alSourcePropertyF(al.AL_ROLLOFF_FACTOR)
    reference_distance = alSourcePropertyF(al.AL_REFERENCE_DISTANCE)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    state = alSourcePropertyI(al.AL_SOURCE_STATE)
    stateToString = {
        al.AL_INITIAL: 'initial',
        al.AL_PLAYING: 'playing',
        al.AL_PAUSED: 'paused',
        al.AL_STOPPED: 'stopped',
        }

    looping = alSourcePropertyI(al.AL_LOOPING)
    type = alSourcePropertyI(al.AL_SOURCE_TYPE)

    buffer = alSourcePropertyI(al.AL_BUFFER)
    buffers_queued = alSourcePropertyI(al.AL_BUFFERS_QUEUED)
    buffers_processed = alSourcePropertyI(al.AL_BUFFERS_PROCESSED)

    if hasattr(al, 'AL_SEC_OFFSET'):
        # OpenAL 1.1 addition
        sec_offset = alSourcePropertyF(al.AL_SEC_OFFSET)
        sample_offset = alSourcePropertyI(al.AL_SAMPLE_OFFSET)
        byte_offset = alSourcePropertyI(al.AL_BYTE_OFFSET)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, buffer=None, bPlay=True, bCreate=True):
        if not bCreate:
            return

        self.create()
        if buffer is not None:
            self.queue(buffer)
            if bPlay:
                self.play()

    def __repr__(self):
        return "<%s.%s alid: %s state: %s>" % (
                self.__class__.__module__,
                self.__class__.__name__,
                self._getALID(True), self.getStateStr())

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def isSource(self):
        return True

    def create(self):
        if self._hasALID():
            raise Exception("Source has already been created")

        sourceIds = (1*al.ALuint)()
        al.alGenSources(1, sourceIds)
        self.createFromId(sourceIds[0])

    def createFromId(self, sourceId):
        self._alid_ = sourceId
        self.getObjContext().addSource(self)

    def destroy(self):
        if self._hasALID():
            i = self.withObjContext()
            try:
                self.destroyFromId(self._alid_)
            finally:
                del self._alid_
                i.next()

    def destroyFromId(self, sourceId):
        self.stop(True)

        sourceIds = (1*al.ALuint)()
        sourceIds[:] = [sourceId]
        al.alDeleteSources(1, sourceIds)

        ctx = self.getObjContext()
        if ctx is not None:
            ctx.removeSource(self)

    _bufferQueue = None
    def getBufferQueue(self):
        if self._bufferQueue is None:
            self._bufferQueue = []
        return self._bufferQueue

    def isQueued(self, buffer):
        return (buffer in self.getBufferQueue())
    def __contains__(self, buffer):
        return self.isQueued(buffer)

    def queue(self, *buffers):
        if len(buffers) == 1:
            if isinstance(buffers[0], (list, tuple)):
                buffers = buffers[0]

        bufids = (al.ALuint * len(buffers))()
        bufids[:] = [buf._alid_ for buf in buffers]
        al.alSourceQueueBuffers(self._alid_, len(bufids), bufids)
        for buf in buffers:
            self.onQueueBuffer(buf)
    enqueue = queue

    def dequeue(self, *buffers):
        if len(buffers) == 1:
            if isinstance(buffers[0], (list, tuple)):
                buffers = buffers[0]

        buffers = [buf for buf in buffers if buf in self.getBufferQueue()]
        bufids = (al.ALuint * len(buffers))()
        bufids[:] = [buf._alid_ for buf in buffers]
        al.alSourceUnqueueBuffers(self._alid_, len(bufids), bufids)

        for buf in buffers:
            self.onDequeueBuffer(buf)
    unqueue = dequeue

    def dequeueAll(self):
        self.dequeue(self.getBufferQueue())

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def onQueueBuffer(self, buf):
        buf.onQueued(self)
        self.getBufferQueue().append(buf)

    def onDequeueBuffer(self, buf):
        buf.onDequeued(self)
        self.getBufferQueue().remove(buf)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getQueue(self):
        return self.getBufferQueue()
    def setQueue(self, *buffers):
        self.stop(True)
        self.queue(*buffers)

    def playQueue(self, *buffers):
        self.setQueue(*buffers)
        self.play()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def play(self, *buffers):
        if buffers:
            self.queue(*buffers)
        al.alSourcePlay(self._alid_)
    def pause(self):
        al.alSourcePause(self._alid_)
    def stop(self, dequeueAll=True):
        al.alSourceStop(self._alid_)
        if dequeueAll:
            self.dequeueAll()
    def rewind(self):
        al.alSourceRewind(self._alid_)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def playFile(self, anAudioFile):
        from TG.openAL.buffer import Buffer
        self.playQueue(Buffer.fromFile(anAudioFile))

    def playFilename(self, anAudioFilename):
        from TG.openAL.buffer import Buffer
        self.playQueue(Buffer.fromFilename(anAudioFilename))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getStateStr(self):
        if self._hasALID():
            return self.stateToString[self.state]
        else: return "uninitialized"
    
    def isPlaying(self):
        return self.state == al.AL_PLAYING
    def isPaused(self):
        return self.state == al.AL_PAUSED
    def isStopped(self):
        return self.state in (al.AL_STOPPED, al.AL_INITIAL)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class SourceCollection(object):
    def __init__(self, sources=None):
        if isinstance(sources, (int, long)):
            self.setSources(Source() for s in xrange(sources))
        elif sources is not None:
            self.setSources(sources)

    _sources = None
    def getSources(self):
        if self._sources is None:
            self.setSources([])
        return self._sources
    def setSources(self, sources):
        sources = list(sources)
        for s in sources:
            if not s.isSource():
                raise ValueError('All entries must be source objects')
        self._sources = sources

    def getSourceIds(self):
        srcids = (al.ALuint * len(self))()
        srcids[:] =  [s._alid_ for s in self.getSources()]
        return srcids

    def __len__(self):
        return self.getSources().__len__()
    def __iter__(self):
        return iter(self.getSources())
    def __getitem__(self, idx):
        return self.getSources().__getitem__(idx)
    def __setitem__(self, idx, value):
        self.getSources().__setitem__(idx, value)
    def __delitem__(self, idx):
        self.getSources().__delitem__(idx)
    def __getslice__(self, slice):
        return self.getSources().__getslice__(slice)
    def __setslice__(self, slice, value):
        self.getSources().__getslice__(slice, value)
    def __delslice__(self, slice):
        self.getSources().__delslice__(slice)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def play(self):
        srcids = self.getSourceIds()
        al.alSourcePlayv(len(srcids), srcids)
    def pause(self):
        srcids = self.getSourceIds()
        al.alSourcePausev(len(srcids), srcids)
    def stop(self):
        srcids = self.getSourceIds()
        al.alSourceStopv(len(srcids), srcids)
    def rewind(self):
        srcids = self.getSourceIds()
        al.alSourceRewindv(len(srcids), srcids)
    
    def isPlaying(self):
        for s in self.getSources():
            if s.isPlaying():
                return True
        else: return False
    def isPaused(self):
        return not self.isStopped() and not self.isPlaying()
        for s in self.getSources():
            if s.isPlaying():
                return True
        else: return False
    def isStopped(self):
        for s in self.getSources():
            if not s.isStopped():
                return False
        return True


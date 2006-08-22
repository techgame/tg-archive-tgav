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

class _alListenerPropertyF(alBasicProperty):
    apiType = al.ALfloat
    apiGet = al.alGetListenerf
    apiSet = al.alListenerf

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class _alListenerPropertyFV(alVectorProperty):
    enumToCount = {
        al.AL_POSITION: 3,
        al.AL_VELOCITY: 3,
        al.AL_ORIENTATION: 6,
    }
    apiType = al.ALfloat
    apiGet = al.alGetListenerfv
    apiSet = al.alListenerfv

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Listener(ALContextObject):
    gain = _alListenerPropertyF(al.AL_GAIN)
    position = _alListenerPropertyFV(al.AL_POSITION)
    velocity = _alListenerPropertyFV(al.AL_VELOCITY)
    orientation = _alListenerPropertyFV(al.AL_ORIENTATION)

    def __init__(self, bCreate=True):
        if not bCreate:
            return
        self.create()

    def __del__(self):
        self.destroy()

    def create(self):
        self._captureCurrentContext()

    def destroy(self):
        pass

    def getDirection(self):
        return self.orientation[0:3]
    def setDirection(self, direction, up=None):
        i = self.inContext()
        try:
            up = list(up or self.getUp())
            self.orientation = list(direction) + up
        finally:
            i.next()
    direction = property(getDirection, setDirection)

    def getUp(self):
        return self.orientation[3:6]
    def setUp(self, up, direction=None):
        direction = list(direction or self.getDirection())
        self.orientation = direction + list(up)
    up = property(getUp, setUp)



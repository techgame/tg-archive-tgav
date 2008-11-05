##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2007  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from ctypes import byref, c_float, c_int, c_char_p, c_void_p
from .base import CepstralObject, _swift, CepstralError

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralEvent(CepstralObject):
    _nameLookup = _swift.swift_event_t.lookup.copy()
    _nameLookup.update([(k, v.split("SWIFT_EVENT_")[1].lower())
            for k, v in _nameLookup.items() if isinstance(k,  int)])

    @classmethod
    def fromEvent(klass, evt_param, evtKind, port):
        self = klass.from_param(evt_param)
        self.setKind(evtKind.value)
        self.port = port
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __repr__(self):
        return '<%s %s (0x%04x)>' % (self.__class__.__name__, self.name, self.kind)

    _kind = None
    def getKind(self):
        return self._kind
    def setKind(self, kind):
        if isinstance(kind, basestring):
            name = kind
            kind = self._nameLookup[name]
        else:
            name = self._nameLookup[kind]
        self._kind = kind
        self._name = name
    kind = property(getKind, setKind)

    _name = None
    def getName(self):
        return self._name
    name = property(getName, setKind)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getContext(self):
        async = c_void_p(0)
        _swift.swift_event_get_context(self, self.port, byref(async))
        return async
    context = property(getContext)

    def getTimes(self):
        start = c_float(); length = c_float()
        _swift.swift_event_get_times(self, byref(start), byref(length))
        return (start.value, start.value+length.value)
    times = property(getTimes)

    def getTextPos(self):
        start = c_int(); length = c_int()
        _swift.swift_event_get_textpos(self, byref(start), byref(length))
        return (start.value, start.value+length.value)
    textPos = property(getTextPos)

    def getText(self):
        text = c_char_p()
        _swift.swift_event_get_text(self, byref(text))
        return text.value
    text = property(getText)

    def getAudio(self):
        raise NotImplementedError('TODO')
        _swift.swift_event_get_audio(self, buf, nbytes)
    audio = property(getAudio)

    def getWave(self):
        raise NotImplementedError('TODO')
        _swift.swift_event_get_wave(self, wave, concatenate)
    wave = property(getWave)

    def getError(self, bRaise=False):
        rv = _swift.swift_result_t(0)
        errmsg = c_char_p()
        _swift.swift_event_get_error(self, byref(rv), byref(errmsg))

        error = CepstralError(rv.value, errmsg.value)
        if bRaise:
            raise error
        else: return error
    error = property(getError)

    @staticmethod
    def _closeFromParam(param):
        pass


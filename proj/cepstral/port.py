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

from ctypes import byref, c_void_p

from TG.metaObserving import MetaObservableType, OBKeyedSet

from .base import CepstralObject, _swift
from .voice import CepstralVoice
from .waveform import CepstralWaveform
from .event import CepstralEvent

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralPort(CepstralObject):
    __metaclass__ = MetaObservableType
    events = OBKeyedSet.property()

    _closeFromParam = staticmethod(_swift.swift_port_close)
    def __init__(self, engine=None, async=True, **kw):
        if engine is None:
            engine = CepstralEngine()
        self.engine = engine

        self.async = async

        if engine is not None:
            self.open(engine, **kw)

    def _set_param(self, as_parameter):
        CepstralObject._set_param(self, as_parameter)
        if self._as_parameter_:
            self.hookEvents()

    def open(self, engine, *argparams, **kwparams):
        if self._as_parameter_ is not None:
            raise RuntimeError("Port already open")

        if argparams or kwparams:
            raise NotImplementedError()

        else: params = None

        self._set_param(_swift.swift_port_open(engine, params))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _swift.swift_port_set_param
    _swift.swift_port_set_params

    def getEncoding(self):
        return _swift.swift_port_language_encoding(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Voices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getVoiceList(self, search, order): 
        voiceParamList = self._getVoiceListRaw(search, order)
        return [CepstralVoice.from_param(voice_param) 
                    for voice_param in voiceParamList]

    def _getVoiceListRaw(self, search, order):
        result = []
        voice_param = _swift.swift_port_find_first_voice(self, search, order)
        while voice_param:
            result.append(voice_param)
            voice_param = _swift.swift_port_find_next_voice(self)
        return result

    _voice = None
    def getVoice(self):
        voice_param = _swift.swift_port_get_current_voice(self)
        voice = self._voice
        if voice is None or voice._as_parameter_ != voice_param:
            voice = CepstralVoice.from_param(voice_param)
            self._voice = voice
        return voice
    def setVoice(self, voice):
        if isinstance(voice, basestring):
            _swift.swift_port_set_voice_by_name(voice)
            self._voice = None

        _swift.swift_port_set_voice(self, voice)
        self._voice = voice
        self.events.call_n1('voice', voice)
    voice = property(getVoice, setVoice)

    def setVoiceName(self, voiceName):
        _swift.swift_port_set_voice_by_name(self, voiceName)
        self._voice = None
        self.events.call_n1('voice', None)
    voiceName = property(None, setVoiceName)

    def setVoiceDir(self, voiceDir):
        _swift.swift_port_set_voice_from_dir(self, voiceDir)
        self._voice = None
        self.events.call_n1('voice', None)
    voiceDir = property(None, setVoiceDir)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Port Methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    statusLookup = dict([(k, v.split("SWIFT_STATUS_")[1].lower())
            for k, v in _swift.swift_status_t.lookup.items() 
                if isinstance(k,  int)])

    def status_id(self, async=None):
        if async is None: 
            async = self.async
        return _swift.swift_port_status(self, async)
    def status_val(self, async=None):
        return self.status_id(async).value
    def status(self, async=None):
        return self.statusLookup[self.status_id(async).value]
    state = property(status)

    def wait(self, async=None):
        if async is None: 
            async = self.async
        if self.status_val() > 0:
            _swift.swift_port_wait(self, async)
            return True
    def stop(self, async=None, place=-1):
        if async is None: 
            async = self.async
        if self.status_val() > 0:
            _swift.swift_port_stop(self, async, place)
            return True
    def pause(self, async=None, place=-1):
        if async is None: 
            async = self.async
        if self.status_val() > 0:
            _swift.swift_port_pause(self, async, place)
            return True
    def resume(self, async=None, place=-1):
        if self.status() == 'pause':
            # pause a second time resumes for cepstral
            self.pause(async, place)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Async Config
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _async = None
    def getAsync(self):
        return self._async
    def setAsync(self, async):
        if async:
            self._async = c_void_p(int(async))
        else: self._async = None
    async = property(getAsync, setAsync)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Speak methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def speak(self, text, encoding=None, isfile=False, async=None, params=None):
        if async is None: 
            async = self.async
        if isfile:
            _swift.swift_port_speak_file(self, text, encoding, async, params)
            return 

        if isinstance(text, unicode):
            encoding = encoding or 'utf-8'
            text = text.encode(encoding)

        _swift.swift_port_speak_text(self, text, len(text), encoding, async, params)
        return self
    speakText = speak

    def phonemes(self, text, encoding=None, isfile=False, params=None):
        if isfile:
            result = _swift.swift_port_get_phones(self, text, 0, encoding, isfile, params)
            return result

        if isinstance(text, unicode):
            encoding = encoding or 'utf-8'
            text = text.encode(encoding)

        result = _swift.swift_port_get_phones(self, text, len(text), encoding, isfile, params)
        return result

    def wave(self, text, encoding=None, isfile=False, params=None):
        if isfile:
            waveform_param = _swift.swift_port_get_wave(self, text, 0, encoding, isfile, params)
            return CepstralWaveform.from_param(waveform_param)

        if isinstance(text, unicode):
            encoding = encoding or 'utf-8'
            text = text.encode(encoding)

        waveform_param = _swift.swift_port_get_wave(self, text, len(text), encoding, isfile, params)
        return CepstralWaveform.from_param(waveform_param)

    def playWave(self, wave, async=None, params=None):
        if async is None: 
            async = self.async
        _swift.swift_port_play_wave(self, wave, async, params)
        return self

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Callbacks
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def hookEvents(self, mask=0x7fe):
        self._setCallback(self._onSynthesisEvent, mask)

    def _setCallback(self, fn, mask=0x7fe):
        fn_cb = _swift.swift_callback_t(fn)
        self._fn_cb = fn_cb
        _swift.swift_port_set_callback(self, fn_cb, mask, 32)

    def _onSynthesisEvent(self, evt_param, evtKind, user):
        evt = CepstralEvent.fromEvent(evt_param, evtKind, self)
        self.events.call_n1(evt.name, evt)
        return 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Misc
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # XXX: _swift.swift_port_get_perfstats
    # XXX: _swift.swift_port_load_sfx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from engine import CepstralEngine


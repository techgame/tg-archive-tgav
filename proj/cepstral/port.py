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

from .base import CepstralObject, _swift
from .voice import CepstralVoice
from .waveform import CepstralWaveform

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralPort(CepstralObject):
    _closeFromParam = staticmethod(_swift.swift_port_close)

    def __init__(self, engine, **kw):
        self.engine = engine
        if engine is not None:
            self.open(engine, **kw)

    def open(self, engine, *argparams, **kwparams):
        if self._as_parameter_ is not None:
            raise RuntimeError("Port already open")

        if argparams or kwparams:
            raise NotImplementedError()

        else: params = None

        self._setAsParam(_swift.swift_port_open(engine, params))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Params
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _swift.swift_port_set_callback
    _swift.swift_port_set_param
    _swift.swift_port_set_params

    def getEncoding(self):
        return _swift.swift_port_language_encoding(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Voices
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getVoiceList(self, search, order): 
        voiceParamList = self._getVoiceListRaw(search, order)
        return [CepstralVoice.fromParam(voice_param) 
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
            voice = CepstralVoice.fromParam(voice_param)
            self._voice = voice
        return voice
    def setVoice(self, voice):
        if isinstance(voice, basestring):
            _swift.swift_port_set_voice_by_name(voice)
            self._voice = None

        _swift.swift_port_set_voice(self, voice)
        self._voice = voice
    voice = property(getVoice, setVoice)

    def setVoiceName(self, voiceName):
        _swift.swift_port_set_voice_by_name(voiceName)
        self._voice = None
    def setVoiceDir(self, voiceDir):
        _swift.swift_port_set_voice_from_dir(voiceDir)
        self._voice = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Port Methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def status(self, async=None):
        _swift.swift_port_status(self, async)

    def wait(self, async=None):
        _swift.swift_port_wait(self, async)
    def stop(self, async=None, place=None):
        _swift.swift_port_stop(self, async, place)
    def pause(self, async=None, place=None):
        _swift.swift_port_pause(self, async, place)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Speak methods
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def speak(self, text, encoding=None, isfile=False, async=None, params=None):
        if isfile:
            _swift.swift_port_speak_file(self, text, encoding, async, params)
            return 

        if isinstance(text, unicode):
            encoding = encoding or 'utf-8'
            text = text.encode(encoding)

        _swift.swift_port_speak_text(self, text, len(text), encoding, async, params)
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
            return CepstralWaveform.fromParam(waveform_param)

        if isinstance(text, unicode):
            encoding = encoding or 'utf-8'
            text = text.encode(encoding)

        waveform_param = _swift.swift_port_get_wave(self, text, len(text), encoding, isfile, params)
        return CepstralWaveform.fromParam(waveform_param)

    def playWave(self, wave, async=None, params=None):
        _swift.swift_port_play_wave(self, wave, async, params)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # XXX: _swift.swift_port_get_perfstats
    # XXX: _swift.swift_port_load_sfx


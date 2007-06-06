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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralEngine(CepstralObject):
    _closeFromParam = staticmethod(_swift.swift_engine_close)

    def __init__(self, **kw):
        if kw.pop('open', True):
            self.open(**kw)

    def open(self, *argparams, **kwparams):
        if self._as_parameter_ is not None:
            raise RuntimeError("Engine already open")

        if argparams or kwparams:
            raise NotImplementedError()
        else: params = None

        self._set_param(_swift.swift_engine_open(params))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def newPort(self, *argparams, **kwparams):
        return CepstralPort(self, *argparams, **kwparams)

    _port = None
    def getPort(self):
        port = self._port
        if port is None:
            port = self.newPort()
            self._port = port
        return port
    port = property(getPort)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _voiceRententionPolicyMap = {
        'forever': _swift.swift_voice_retention_policy_t.SWIFT_VOICE_RETAIN_FOREVER,
        'port': _swift.swift_voice_retention_policy_t.SWIFT_VOICE_RETAIN_PORT_USAGE,
        'none': _swift.swift_voice_retention_policy_t.SWIFT_VOICE_RETAIN_NONE,
        }
    _voiceRententionPolicyMap.update([(v,k) for k,v in _voiceRententionPolicyMap.items()])
    _voiceRententionPolicyMap.update(_swift.swift_voice_retention_policy_t.lookup)

    def getVoiceRententionPolicy(self):
        policy = _swift.swift_engine_get_voice_retention_policy(self)
        return self._voiceRententionPolicyMap[policy]
    def setVoiceRententionPolicy(self, value):
        value = self._voiceRententionPolicyMap[value]
        _swift.swift_engine_set_voice_retention_policy(self, value)
    voiceRententionPolicy = property(getVoiceRententionPolicy, setVoiceRententionPolicy)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from .port import CepstralPort


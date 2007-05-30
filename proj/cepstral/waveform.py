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

class CepstralWaveform(CepstralObject):
    _closeFromParam = staticmethod(_swift.swift_waveform_close)

    def create(self):
        if self._as_parameter_ is not None:
            raise RuntimeError("Waveform already initialized")

        self._setAsParam(_swift.swift_waveform_new())

    def open(self, filename, format, encoding, sample_rate, num_channels):
        if self._as_parameter_ is not None:
            raise RuntimeError("Waveform already open")

        waveform_param = _swift.swift_waveform_open(filename, format, encoding, sample_rate, num_channels)
        self._setAsParam(waveform_param)

    def save(self, filename, format):
        _swift.swift_waveform_save(self, filename, format)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def concat(self, other):
        _swift.swift_waveform_concat(self, other)
    append = concat

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def printWave(self):
        _swift.swift_waveform_print(self)

    def setChannels(self, channels):
        _swift.swift_waveform_set_channels(self, channels)

    def getEncoding(self):
        return _swift.swift_waveform_get_encoding(self)
    def setEncoding(self, encoding):
        _swift.swift_waveform_convert(self, encoding)
    convert = setEncoding
    encoding = property(getEncoding, setEncoding)

    def getSampleRate(self):
        return _swift.swift_waveform_get_sps(self)
    def setSampleRate(self, sampleRate):
        _swift.swift_waveform_resample(self, sampleRate)
    rate = sps = property(getSampleRate, setSampleRate)
    resample = setSampleRate

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getSamples(self, ):
        raise NotImplementedError('Todo Responsibility: %r' % (self,))
        _swift.swift_waveform_get_samples(self, samples, len(samples), bytes_per_sample)
        return samples

    def setSamples(self, samples, frequency, channels):
        raise NotImplementedError('Todo Responsibility: %r' % (self,))
        _swift.swift_waveform_set_samples(samples, len(samples), frequency, channels)


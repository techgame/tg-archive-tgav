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

import audioCodecs
from riffChunk import RIFFReadChunk, RIFFWriteChunk

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveFormatEnum:
    WAVE_FORMAT_UNKNOWN = 0x0000
    WAVE_FORMAT_PCM = 0x0001                # Microsoft Corporation
    WAVE_FORMAT_ADPCM = 0x0002              # Microsoft Corporation
    WAVE_FORMAT_IBM_CVSD = 0x0005           # IBM Corporation
    WAVE_FORMAT_ALAW = 0x0006               # Microsoft Corporation
    WAVE_FORMAT_MULAW = 0x0007              # Microsoft Corporation
    WAVE_FORMAT_OKI_ADPCM = 0x0010          # OKI
    WAVE_FORMAT_IMA_ADPCM = 0x0011          # Intel Corporation
    WAVE_FORMAT_DVI_ADPCM = 0x0011          # Intel Corporation
    WAVE_FORMAT_MEDIASPACE_ADPCM = 0x0012   # Videologic
    WAVE_FORMAT_SIERRA_ADPCM = 0x0013       # Sierra Semiconductor Corp
    WAVE_FORMAT_G723_ADPCM = 0x0014         # Antex Electronics Corporation         
    WAVE_FORMAT_DIGISTD = 0x0015            # DSP Solutions, Inc.
    WAVE_FORMAT_DIGIFIX = 0x0016            # DSP Solutions, Inc.
    WAVE_FORMAT_YAMAHA_ADPCM = 0x0020       # Yamaha Corporation of America
    WAVE_FORMAT_SONARC = 0x0021             # Speech Compression
    WAVE_FORMAT_DSPGROUP_TRUESPEECH = 0x0022# DSP Group, Inc
    WAVE_FORMAT_ECHOSC1 = 0x0023            # Echo Speech Corporation
    WAVE_FORMAT_AUDIOFILE_AF36 = 0x0024     # Audiofile, Inc.
    WAVE_FORMAT_APTX = 0x0025               # Audio Processing Technology
    WAVE_FORMAT_AUDIOFILE_AF10 = 0x0026     # Audiofile, Inc.
    WAVE_FORMAT_GSM610 = 0x0031             # Microsoft Corporation
    WAVE_FORMAT_DOLBY_AC2 = 0x0030          # Dolby Laboratories
    WAVE_FORMAT_CONTROL_RES_VQLPC = 0x0034  # Control Resources Limited
    WAVE_FORMAT_ANTEX_ADPCME = 0x0033       # Antex Electronics Corporation         
    WAVE_FORMAT_DIGIREAL = 0x0035           # DSP Solutions, Inc.
    WAVE_FORMAT_DIGIADPCM = 0x0036          # DSP Solutions, Inc.
    WAVE_FORMAT_CONTROL_RES_CR10 = 0x0037   # Control Resources Limited
    WAVE_FORMAT_NMS_VBXADPCM = 0x0038       # Natural MicroSystems
    WAVE_FORMAT_G721_ADPCM = 0x0040         # Antex Electronics Corporation         
    WAVE_FORMAT_MPEG = 0x0050               # Microsoft Corporation
    WAVE_FORMAT_FM_TOWNS_SND = 0x0300       # Fujitsu Corp.
    WAVE_FORMAT_CREATIVE_ADPCM = 0x0200     # Creative Labs, Inc
    WAVE_FORMAT_OLIGSM = 0x1000             # Ing C. Olivetti & C., S.p.A.
    WAVE_FORMAT_OLIADPCM = 0x1001           # Ing C. Olivetti & C., S.p.A.
    WAVE_FORMAT_OLICELP = 0x1002            # Ing C. Olivetti & C., S.p.A.
    WAVE_FORMAT_OLISBC = 0x1003             # Ing C. Olivetti & C., S.p.A.
    WAVE_FORMAT_OLIOPR = 0x1004             # Ing C. Olivetti & C., S.p.A.

    mapToName = {
        None: 'Unknown',
        WAVE_FORMAT_UNKNOWN: 'Unknown',
        WAVE_FORMAT_PCM: 'PCM',
        WAVE_FORMAT_ADPCM: 'ADPCM',
        WAVE_FORMAT_IBM_CVSD: 'IBM CVSD',
        WAVE_FORMAT_ALAW: 'ALAW',
        WAVE_FORMAT_MULAW: 'MULAW',
        WAVE_FORMAT_OKI_ADPCM: 'OKI ADPCM',
        WAVE_FORMAT_DVI_ADPCM: 'DVI ADPCM',
        #WAVE_FORMAT_IMA_ADPCM: 'IMA ADPCM',
        WAVE_FORMAT_MEDIASPACE_ADPCM: 'Mediaspace ADPCM',
        WAVE_FORMAT_SIERRA_ADPCM: 'Sierra ADPCM',
        WAVE_FORMAT_G723_ADPCM: 'G723 ADPCM',
        WAVE_FORMAT_DIGISTD: 'Digi STD',
        WAVE_FORMAT_DIGIFIX: 'Digi FIX',
        WAVE_FORMAT_YAMAHA_ADPCM: 'YAMAHA ADPCM',
        WAVE_FORMAT_SONARC: 'SONARC',
        WAVE_FORMAT_DSPGROUP_TRUESPEECH: 'DSPGroup Truespeech',
        WAVE_FORMAT_ECHOSC1: 'ECHOSC1',
        WAVE_FORMAT_AUDIOFILE_AF36: 'Audiofile AF36',
        WAVE_FORMAT_APTX: 'APTX',
        WAVE_FORMAT_AUDIOFILE_AF10: 'Audiofile AF10',
        WAVE_FORMAT_GSM610: 'GSM610',
        WAVE_FORMAT_DOLBY_AC2: 'Dolby AC2',
        WAVE_FORMAT_CONTROL_RES_VQLPC: 'Control Res VQLPC',
        WAVE_FORMAT_ANTEX_ADPCME: 'Antex ADPCME',
        WAVE_FORMAT_DIGIREAL: 'Digi REAL',
        WAVE_FORMAT_DIGIADPCM: 'Digi ADPCM',
        WAVE_FORMAT_CONTROL_RES_CR10: 'Control Res CR10',
        WAVE_FORMAT_NMS_VBXADPCM: 'NMS VBX ADPCM',
        WAVE_FORMAT_G721_ADPCM: 'G721 ADPCM',
        WAVE_FORMAT_MPEG: 'MPEG',
        WAVE_FORMAT_FM_TOWNS_SND: 'FM Towns SND',
        WAVE_FORMAT_CREATIVE_ADPCM: 'Creative ADPCM',
        WAVE_FORMAT_OLIGSM: 'OLI GSM',
        WAVE_FORMAT_OLIADPCM: 'OLI ADPCM',
        WAVE_FORMAT_OLICELP: 'OLI CELP',
        WAVE_FORMAT_OLISBC: 'OLI SBC',
        WAVE_FORMAT_OLIOPR: 'OLI OPR',
    }
    mapFromName = {}
    mapFromName.update((x,x) for x in mapToName.iterkeys())
    mapFromName.update((y,x) for x,y in mapToName.iteritems())
    mapFromName.update((y.lower(),x) for x,y in mapToName.iteritems())
    mapFromName[None] = WAVE_FORMAT_PCM

    @classmethod
    def _asDict(klass, inverse=False):
        if inverse:
            f = lambda x,y: (y,x)
        else: 
            f = lambda x,y: (x,y)

        result = dict(f(x,y) for x,y in vars(WaveFormatEnum).iteritems() if x.startswith('WAVE_FORMAT'))
        result[None] = klass.WAVE_FORMAT_UNKNOWN
        return result

waveFormatToName = WaveFormatEnum.mapToName
globals().update(WaveFormatEnum._asDict())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wave RIFF Formats
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveReadChunk(RIFFReadChunk):
    align = 2
    packEndian = '<'
    _chunkHeaderFmt = '4sL'
    _chunkHeaderSize = 8
    _excludeHeaderSize = True

class WaveWriteChunk(RIFFWriteChunk):
    align = 2
    packEndian = '<'
    _chunkHeaderFmt = '4sL'
    _chunkHeaderSize = 8
    _excludeHeaderSize = True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wave Codec Stream Reader/Writer
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveCodecStream(audioCodecs.CodecStream):
    @classmethod
    def codecNameFromFormat(klass, format):
        return '%s (0x%x)' % (waveFormatToName.get(format, 'Unknown'), format or 0)

    def readWaveHeader(self, wave, hostStream, close=False):
        return self.readRIFFChunk(wave, hostStream, close)

    def readRIFFChunk(self, wave, hostStream, close=False):
        riff = WaveReadChunk(hostStream)
        if riff.getChunkName() != 'RIFF':
            raise Exception("Invalid RIFF header")

        riffType = riff.read(4)
        if riffType != 'WAVE':
            raise Exception("RIFF header specifies a '%s' file, not 'WAVE'" % riffType)

        self._readSubChunks(wave, riff)

        if close:
            self.close()
        return riff

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _readSubChunks(self, wave, riff):
        readUnknownChunk = self.__class__.readUnknownChunk
        try:
            while 1:
                chunk = riff.newRIFFChunk()
                if not chunk: 
                    break

                chunkDispatch = self._readDispatchMap.get(chunk.getChunkName(), readUnknownChunk)
                chunkDispatch(self, wave, chunk)

                chunk.skip()
        except StopIteration:
            pass

    _readDispatchMap = {}
    def _addChunkDispatch_(key, _readDispatchMap=_readDispatchMap):
        if key:
            key = key[:4].ljust(4)

        def decorate(func):
            _readDispatchMap[key] = func
            return func
        return decorate

    @_addChunkDispatch_('fmt ')
    def readFmtChunk(self, wave, fmtChunk):
        self._unpackFmtChunkCommon(wave, fmtChunk.readAndUnpack)
        self._unpackFmtChunkExtra(wave, fmtChunk.readAndUnpack)

    def _unpackFmtChunkCommon(self, wave, unpack):
        fmtHeader = unpack('HHLLHH')

        (self.format, self.channels, self.frequency) = fmtHeader[:3]
        (wave.waveFormat, wave.channels, wave.frequency) = fmtHeader[:3]

        (avgBytesPerSec, blockAlign, bitsPerSample) = fmtHeader[3:7]
        wave.setBitWidth(bitsPerSample)
        self.setBitWidth(bitsPerSample)

    def _unpackFmtChunkExtra(self, wave, pack):
        try:
            codec = self.getCodec()
        except KeyError:
            return
        return codec._packFmtChunkExtra(wave, pack)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @_addChunkDispatch_('fact')
    def readFactChunk(self, wave, factChunk):
        pcmFrames = factChunk.readAndUnpack('L')[0]

    @_addChunkDispatch_('data')
    def readDataChunk(self, wave, dataChunk):
        self.setStream(dataChunk)

    @_addChunkDispatch_('LIST')
    def readListChunk(self, wave, listChunk):
        pass

    @_addChunkDispatch_(None)
    @_addChunkDispatch_('')
    def readUnknownChunk(self, wave, listChunk):
        pass

    del _addChunkDispatch_

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def writeWaveHeader(self, wave, hostStream, close=False):
        return self.writeRIFFChunk(wave, hostStream, close)

    def writeRIFFChunk(self, wave, hostStream, close=False):
        riff = WaveWriteChunk(hostStream, 'RIFF')
        riff.write('WAVE')
        fmt = self.writeFmtChunk(wave, riff)
        fact = self.writeFactChunk(wave, riff)
        data = self.writeDataChunk(wave, riff)
        if close:
            riff.close()
        return riff

    def writeFmtChunk(self, wave, riff):
        fmtChunk = riff.getRIFFChunkOrNew('fmt ')
        fmtData = self._packFmtChunkCommon(wave, fmtChunk.pack)
        fmtData += self._packFmtChunkExtra(wave, fmtChunk.pack)
        fmtChunk.writeAll(fmtData)
        return fmtData

    def _packFmtChunkCommon(self, wave, pack):
        return pack('HHLLHH', 
                wave.waveFormat, 
                wave.channels, 
                wave.frequency, 
                wave.getAvgBytesPerSec(),
                wave.getBlockAlign(),
                wave.getBitWidth())

    def _packFmtChunkExtra(self, wave, pack):
        try:
            codec = self.getCodec()
        except KeyError:
            pass
        return codec._packFmtChunkExtra(wave, pack)

    def writeFactChunk(self, wave, riff):
        if self.format != WAVE_FORMAT_PCM:
            factChunk = riff.getRIFFChunkOrNew('fact')
            factData = self._packFactChunk(wave, factChunk.pack)
            factChunk.writeAll(factData)
            return factChunk

    def _packFactChunk(self, wave, pack):
        if self.format != WAVE_FORMAT_PCM:
            return pack('L', self.getSamplesWritten())

    def writeDataChunk(self, wave, riff):
        dataChunk = riff.getRIFFChunkOrNew('data')
        dataChunk.writeHeader()

        if self.getRawWritten():
            rawStream = self.getStream()
            rawStream.seek(0)

            while 1:
                data = rawStream.read(8192)
                if data:
                    dataChunk.write(data)
                else: 
                    break

        self.setStream(dataChunk)
        return dataChunk

WaveCodecStream.newCodecRegistry()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wave specific Codecs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WavePCMCodec(audioCodecs.PCMCodec):
    format = WAVE_FORMAT_PCM

    def _packFmtChunkExtra(self, wave, pack):
        return ''

WaveCodecStream.registerCodec(WavePCMCodec)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveULAWCodec(audioCodecs.ULAWCodec):
    format = WAVE_FORMAT_MULAW

    def _packFmtChunkExtra(self, wave, pack):
        return ''

WaveCodecStream.registerCodec(WaveULAWCodec)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveADPCMCodec(audioCodecs.ADPCMCodec):
    # Which one does audioop create?
    #format = WAVE_FORMAT_DVI_ADPCM
    format = WAVE_FORMAT_ADPCM

    def _packFmtChunkExtra(self, wave, pack):
        raise NotImplementedError('This needs some more work')

WaveCodecStream.registerCodec(WaveADPCMCodec)

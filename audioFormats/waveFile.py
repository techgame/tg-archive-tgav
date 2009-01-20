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

from waveCodecs import WaveCodecStream, WaveFormatEnum

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

globals().update(WaveFormatEnum._asDict())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Wave File
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class WaveFormat(object):
    CodecStreamFactory = WaveCodecStream

    formatEnum = WaveFormatEnum
    waveFormat = WaveFormatEnum.WAVE_FORMAT_PCM
    frequency = 0 #44100
    channels = 0 #2
    bitWidth = 0 #1

    def __init__(self, fileOrName=None, *args, **kw):
        if fileOrName:
            self.open(fileOrName, *args, **kw)

    def __del__(self):
        self.close()

    def __repr__(self):
        return '<%s.%s format: %s frequency: %s channels: %s width: %s>' % (
                    self.__class__.__module__, self.__class__.__name__, 
                    self.getFormatName(), 
                    self.frequency,
                    self.channels,
                    self.width,
                    )

    _file = None
    _riff = None
    def open(self, fileOrName, mode='rb', *args, **kw):
        if isinstance(fileOrName, (str, unicode)):
            fileOrName = file(fileOrName, mode, *args, **kw)
        self._file = fileOrName

        if 'r' in self._file.mode:
            self.readWaveHeader()

    def close(self, closeFile=True):
        if self._riff:
            self._riff.close()
            del self._riff

        if self._file:
            if closeFile:
                self._file.close()
                del self._file
            else:
                self._file.flush()

    def readWaveHeader(self):
        if self._riff is not None:
            return
        codecStream = self.getCodecStream()
        self._riff = codecStream.readWaveHeader(self, self._file, False)

    def writeWaveHeader(self):
        if self._riff is not None:
            return

        codecStream = self.getCodecStream()
        self._riff = codecStream.writeWaveHeader(self, self._file, False)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def copyFormat(self, other, *args, **kw):
        if other:
            self.waveFormat = WaveFormatEnum.mapFromName[other.waveFormat]
            self.frequency = other.frequency
            self.channels = other.channels
            self.setBitWidth(other.bitWidth)
        self.setFormat(*args, **kw)

    def setFormat(self, waveFormat=None, frequency=None, channels=None, width=None, bitWidth=None):
        if waveFormat is not None:
            self.waveFormat = WaveFormatEnum.mapFromName[waveFormat]
        if frequency is not None:
            self.frequency = frequency
        if channels is not None:
            self.channels = channels

        if width is not None:
            self.setByteWidth(width)
        elif bitWidth is not None:
            self.setBitWidth(bitWidth)

    def setFormatFrom(self, waveFormat=None, source=None):
        if source is not None:
            self.setFormat(waveFormat, source.frequency, source.channels, getattr(source, 'width', None), getattr(source, 'bitWidth', None))
        else: self.setFormat(waveFormat)
        
    def getFormatName(self, waveFormat=-1):
        if waveFormat < 0:
            waveFormat = self.waveFormat
        return self.CodecStreamFactory.codecNameFromFormat(waveFormat)

    _codecStream = None
    def getCodecStream(self, encodedStream=None):
        if self._codecStream is None:
            self._codecStream = self.CodecStreamFactory(self.waveFormat, self.channels, self.bitWidth)
        if encodedStream is not None:
            self._codecStream.setStream(encodedStream)
        return self._codecStream

    def getCodec(self):
        return self.getCodecStream().getCodec()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getAvgBytesPerSec(self):
        return self.frequency * self.getBlockAlign()
    def getBlockAlign(self):
        blockAlign = self.channels * self.getBitWidth()
        blockAlign, part = divmod(blockAlign, 8)
        blockAlign += bool(part)
        return blockAlign

    def getBitWidth(self):
        return self.bitWidth
    def setBitWidth(self, bitWidth):
        self.bitWidth = bitWidth

    def getByteWidth(self):
        return self.getBitWidth() // 8
    def setByteWidth(self, byteWidht):
        return self.setBitWidth(byteWidht * 8)
    width = byteWidth = property(getByteWidth)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def tellFrames(self):
        return self.getCodecStream().tell()
    tell = tellFrames

    def seekFrames(self, posFrames, whence=0):
        return self.getCodecStream().seek(posFrames, whence)
    seek = seekFrames

    def rewind(self):
        self.seekFrames(0)

    def writeFrames(self, pcmFrames):
        if self._riff is None:
            self.writeWaveHeader()
        #self._riff.getRIFFChunk('data').syncPosition()
        return self.getCodecStream().write(pcmFrames)
    write = writeframes = writeFrames

    def readFrames(self, count=-1):
        if self._riff is None:
            self.readWaveHeader()
        self._riff.getRIFFChunk('data').syncPosition()
        return self.getCodecStream().read(count)
    read = readframes = readFrames



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

import sys
import array
import audioop

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CodecStream(object):
    format = None
    channels = None
    bitWidth = None

    def __init__(self, format, channels, bitWidth, stream=None):
        self.format = format
        self.channels = channels
        self.bitWidth = bitWidth

        if stream is not None:
            self.setStream(stream)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Codec Registry
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @classmethod
    def newCodecRegistry(klass):
        klass._codecFactoryMap = {}

    @classmethod
    def registerCodec(klass, codec):
        klass.registerCodecByFormat(codec.format, codec)
    @classmethod
    def registerCodecByFormat(klass, format, codecFactory):
        klass._codecFactoryMap[format] = codecFactory
    @classmethod
    def codecFromFormat(klass, format, *args, **kw):
        try:
            factory = klass._codecFactoryMap[format]
        except KeyError:
            raise KeyError("No register codec to support \"%s\" format" % (self.codecNameFromFormat(format),))
        else:
            return factory(*args, **kw)

    @classmethod
    def codecNameFromFormat(klass, format):
        return "Unknown (%s)" % (format,)

    _codec = None
    def getCodec(self, orCreate=True):
        if self._codec is None and orCreate:
            self._codec = self.codecFromFormat(self.format, self.bitWidth)
        return self._codec
    def setCodec(self, codec):
        self._codec = codec
    def delCodec(self):
        del self._codec

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _stream = None
    def getStream(self):
        return self._stream
    def setStream(self, stream):
        self._stream = stream
    def delStream(self):
        del self._stream

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def readStream(self, *args, **kw):
        stream = self.getStream()
        return stream.read(*args, **kw)
    def writeStream(self, fragment):
        stream = self.getStream()
        return stream.write(fragment)

    def tellStream(self):
        stream = self.getStream()
        return stream.tell()
    def seekStream(self, pos, whence):
        stream = self.getStream()
        return stream.seek(pos, whence)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def framePosFromBytes(self, countBytes):
        if countBytes >= 0:
            return self.getCodec().framePosFromBytes(countBytes, self.channels)
        else: return countBytes
    def bytePosFromFrames(self, countFrames):
        if countFrames >= 0:
            return self.getCodec().bytePosFromFrames(countFrames, self.channels)
        else: return countFrames

    def tell(self):
        return self.framePosFromBytes(self.tellStream())
    def seek(self, posFrames, whence=0):
        return self.seekStream(self.bytePosFromFrames(posFrames), whence)

    _readFrames = 0
    _readRaw = 0
    def read(self, countFrames=-1):
        codec = self.getCodec()
        count = self.bytePosFromFrames(countFrames)

        fragment = self.readStream(count)
        self._readRaw += len(fragment)
        pcmFrames = codec.decode(fragment)

        self._readFrames += len(pcmFrames) // self.getFrameWidth()
        return pcmFrames

    _writtenFrames = 0
    _writtenRaw = 0
    def write(self, pcmFrames):
        if not pcmFrames:
            return 0

        codec = self.getCodec()

        fragment = codec.encode(pcmFrames)
        result = self.writeStream(fragment)
        self._writtenFrames += len(pcmFrames) // self.getFrameWidth()
        self._writtenRaw += len(fragment)
        return result

    def getRawRead(self):
        return self._readRaw
    def getRawWritten(self):
        return self._writtenRaw

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getFramesRead(self):
        return self._readFrames
    def getFramesWritten(self):
        return self._writtenFrames
    def getSamplesRead(self):
        return self._readFrames // self.channels
    def getSamplesWritten(self):
        return self._writtenFrames // self.channels

    def getTotalFrames(self):
        return self.getFramesRead() + self.getFramesWritten()

    def getFrameWidth(self):
        bytes, r = divmod(self.channels * self.bitWidth, 8)
        return bytes + bool(r)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
#~ Codecs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Codec(object):
    def __init__(self, bitWidth):
        self.bitWidth = bitWidth
        width = divmod(bitWidth, 8)
        self.width = width[0] + bool(width[1])

    def decode(self, fragment):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def encode(self, frames):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def framePosFromBytes(self, countBytes, channels):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def bytePosFromFrames(self, countFrames, channels):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if sys.byteorder == 'big':
        def _endianness(self, pcmFrames):
            sizefmt = {1: None, 2: 'h', 4: 'l'}[self.width]
            if sizefmt:
                pcmFrames = array.array(sizefmt, pcmFrames)
                pcmFrames.byteswap()
                return pcmFrames.tostring()
            else:
                return pcmFrames
    else:
        def _endianness(self, pcmFrames):
            return pcmFrames

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PCMCodec(Codec):
    def decode(self, pcmFrames):
        return self._endianness(pcmFrames)
    def encode(self, pcmFrames):
        return pcmFrames

    def framePosFromBytes(self, countBytes, channels):
        return countBytes // (self.width * channels)
    def bytePosFromFrames(self, countFrames, channels):
        return countFrames * (self.width * channels)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ULAWCodec(Codec):
    def decode(self, fragment):
        pcmFrames = audioop.ulaw2lin(fragment, self.width)
        return pcmFrames
    def encode(self, pcmFrames):
        fragment = audioop.lin2ulaw(pcmFrames, self.width)
        return fragment

    def framePosFromBytes(self, countBytes, channels):
        return countBytes // channels
    def bytePosFromFrames(self, countFrames, channels):
        return countFrames * channels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ADPCMCodec(Codec):
    encodeState = None
    decodeState = None

    def decode(self, fragment):
        pcmFrames, self.decodeState = audioop.adpcm2lin(fragment, self.width, self.decodeState)
        return pcmFrames
    def encode(self, pcmFrames):
        fragment, self.encodeState = audioop.lin2adpcm(pcmFrames, self.width, self.encodeState)
        return fragment

    def framePosFromBytes(self, countBytes, channels):
        return countBytes * 2 // channels
    def bytePosFromFrames(self, countFrames, channels):
        return (countFrames * channels // 2)


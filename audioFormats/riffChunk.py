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

import struct

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RIFFChunkError(Exception):
    pass

class RIFFChunkIOError(RIFFChunkError):
    pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ RIFF Chunk Commonalities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RIFFBaseChunk(object):
    byteAlign = 2 # bytes
    packEndian = '>' # '>' is big endian, '<' is little endian, '!' is network order, '@' is native order, '=' is native order with std size & alignment
    _chunkHeaderFmt = '4sL'
    _chunkHeaderSize = 8
    _closed = False
    
    def __init__(self, hostStream, byteAlign=NotImplemented, packEndian=NotImplemented):
        # whether to byteAlign to word (2-byte) boundaries
        if byteAlign is not NotImplemented:
            if byteAlign is True:
                byteAlign = 2
            self.byteAlign = int(byteAlign)

        if packEndian is not NotImplemented:
            if packEndian not in '<>!@=':
                raise ValueError("Pack Endian must be a valid struct pack directive")
            self.packEndian = packEndian

        if self._initForStream(hostStream):
            self.setHostStream(hostStream)

    def __repr__(self):
        return '<%s.%s name: %s size: %s align: %s pack: %r>' % (
                    self.__class__.__module__, self.__class__.__name__, 
                    self.getChunkName(), self.getChunkSize(),
                    self.byteAlign, self.packEndian,)

    _hostStream = None
    def getHostStream(self):
        return self._hostStream
    def setHostStream(self, hostStream):
        self._hostStream = hostStream
    def delHostStream(self):
        del self._hostStream

    def _initForStream(self, hostStream):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def pack(self, format, *args):
        format = self.packEndian+format
        return struct.pack(format, *args)
    def unpack(self, format, *args):
        format = self.packEndian+format
        return struct.unpack(format, *args)
    def calcsize(self, format):
        return struct.calcsize(format)

    def readAndUnpack(self, format):
        size = self.calcsize(format)
        rawdata = self.read(size)
        return self.unpack(format, rawdata)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __nonzero__(self):
        return bool(self.getChunkName() or self.getChunkSize()) 

    _chunkName = None
    def getChunkName(self):
        return self._chunkName
    def setChunkName(self, name):
        self._chunkName = name

    _chunkSize = None
    def getChunkSize(self):
        return self._chunkSize
    def setChunkSize(self, size):
        self._chunkSize = size

    def getChunkOffset(self):
        return self.getChunkDataOffset() - self._chunkHeaderSize
    def setChunkOffset(self, chunkOffset):
        return self.setChunkDataOffset(chunkOffset + self._chunkHeaderSize)

    _chunkDataOffset = None
    def getChunkDataOffset(self):
        return self._chunkDataOffset
    def setChunkDataOffset(self, chunkDataOffset):
        self._chunkDataOffset = chunkDataOffset

    _chunkPosition = None
    def _getChunkPosition(self):
        return self._chunkPosition
    def _setChunkPosition(self, chunkPosition):
        self._chunkPosition = chunkPosition
    def _delChunkPosition(self):
        self._chunkPosition = None


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def isatty(self):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")
        return False

    def read(self, count=-1):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))
    def write(self, data):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    def close(self):
        if not self._closed:
            self.finalize()
            self.skip()
            self._closed = True
    
    def finalize(self):
        chunk = None
        for chunk in self.iterRIFFChunks():
            chunk.finalize()

    def isSeekable(self):
        return self.getChunkDataOffset() is not None

    def seek(self, pos, whence=0, updatePos=True, ifAllowed=False):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")

        if not self.isSeekable():
            if ifAllowed:
                return False
            else:
                raise RIFFChunkIOError("RIFF Chunk's host stream is not seekable")

        chunkSize = self.getChunkSize()
        if whence == 1:
            pos += self._getChunkPosition()
        elif chunkSize is not None:
            if whence == 2:
                pos += chunkSize
            pos = min(pos, chunkSize)

        pos = max(0, pos)
        if self.byteAlign is not None:
            pos += (pos % self.byteAlign)

        self._seekHostAbs(self.getChunkDataOffset() + pos)

        if updatePos:
            self._setChunkPosition(pos)

        return True

    def _seekHostAbs(self, hostPos):
        return self.getHostStream().seek(hostPos, 0)

    def syncPosition(self, bHeader=False):
        if bHeader:
            self._delChunkPosition()
            self._seekHostAbs(self.getChunkOffset())
        else: 
            self._seekHostAbs(self.getChunkDataOffset() + self._getChunkPosition())
        return self._getChunkPosition()

    def tellHost(self):
        return self.getHostStream().tell()
    def tellRoot(self):
        hostStream = self.getHostStream()
        while hasattr(hostStream, 'getHostStream'):
            hostStream = hostStream.getHostStream()
        return hostStream.tell()

    def tell(self, fromHost=False):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")

        if fromHost:
            hostStream = self.getHostStream()
            chunkOffset = self.getChunkDataOffset()
            if chunkOffset is not None:
                return hostStream.tell() - chunkOffset
            else:
                raise RIFFChunkIOError("RIFF Chunk's host stream is not seekable")
        else:
            return self._getChunkPosition()
    
    def skip(self):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")

        if not self.seek(0, 2, updatePos=False, ifAllowed=True):
            self._skipUnseekable()
    
    def _skipUnseekable(self):
        raise NotImplementedError('Subclass Responsibility: %r' % (self,))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _chunkList = None
    def addRIFFChunk(self, chunk):
        if self._chunkList is None:
            self._chunkList = []
        elif self._chunkList:
            self._chunkList[-1].finalize()
        self._chunkList.append(chunk)
    
    def getRIFFChunk(self, chunkName):
        for each in self.iterRIFFChunks(chunkName):
            return each
        else: return None

    def newRIFFChunk(self, *args, **kw):
        klass = self.__class__
        return klass(self, *args, **kw)

    def getRIFFChunkOrNew(self, chunkName, *args, **kw):
        result = self.getRIFFChunk(chunkName)
        if result is None:
            result = self.newRIFFChunk(chunkName, *args, **kw)
        return result

    def iterRIFFChunks(self, chunkName=None):
        chunks = self._chunkList or ()
        if chunkName is None:
            return iter(chunks)
        else:
            return (c for c in chunks if (c.getChunkName() == chunkName))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Reading RIFF Chunks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RIFFReadChunk(RIFFBaseChunk):
    def _initForStream(self, hostStream):
        if hasattr(hostStream, 'addRIFFChunk'):
            hostStream.addRIFFChunk(self)

        if not self.readHeader(hostStream):
            return False

        try:
            chunkDataOffset = hostStream.tell()
        except (AttributeError, IOError):
            pass
        else:
            self.setChunkDataOffset(chunkDataOffset)

        self._setChunkPosition(0)
        return True

    def readHeader(self, hostStream):
        name = hostStream.read(4)
        if not name:
            return False

        self.setChunkName(name)

        size = self.unpack('L', hostStream.read(4))[0]
        self.setChunkSize(size)
        return True

    def read(self, size=-1):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")

        #chunkPos = self.syncPosition()
        chunkPos = self._getChunkPosition()
        chunkSize = self.getChunkSize()
        if chunkPos >= chunkSize:
            return ''
        elif chunkPos < 0:
            self.seek(0)
            chunkPos = 0

        remainingSize = max(chunkSize-chunkPos, 0)
        if size < 0:
            # negative size means grab the rest
            size = remainingSize
        else:
            size = min(size, remainingSize)

        hostStream = self.getHostStream()
        data = hostStream.read(size)
        chunkPos += len(data)

        if self.byteAlign is not None:
            if (chunkSize == chunkPos):
                alignmentFiller = hostStream.read(chunkPos % self.byteAlign)
                chunkPos += len(alignmentFiller)

        self._setChunkPosition(chunkPos)
        return data
    
    def write(self, data):
        raise RIFFChunkIOError('RIFF Read Chunk is not writeable')

    def _skipUnseekable(self):
        chunkPos = self.tell()
        if chunkPos < self.getChunkSize():
            while self.read(8192):
                pass

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Writing RIFF Chunks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RIFFWriteChunk(RIFFBaseChunk):
    def __init__(self, hostStream, name=None, data=None, size=None, byteAlign=NotImplemented, packEndian=NotImplemented):
        RIFFBaseChunk.__init__(self, hostStream, byteAlign, packEndian)
        if name:
            self.setChunkName(name)

        size = size or (data and len(data)) or None
        if size:
            self.setChunkSize(size)

        if data:
            self.write(data)

    def _initForStream(self, hostStream):
        if hasattr(hostStream, 'addRIFFChunk'):
            hostStream.addRIFFChunk(self)

        try:
            chunkOffset = hostStream.tell()
        except (AttributeError, IOError):
            pass
        else:
            self.setChunkOffset(chunkOffset)

        return True

    _headerWritten = False
    def writeHeader(self, allowSeek=True):
        hostStream = self.getHostStream()

        if allowSeek:
            offset = self.getChunkOffset()
            if offset != hostStream.tell():
                hostStream.seek(offset)

        name = self.getChunkName()[:4].ljust(4)
        size  = self.getChunkSize() or 0
        header = self.pack(self._chunkHeaderFmt, name, size)

        hostStream.write(header)
        self._setChunkPosition(0)

        if size:
            self._headerWritten = True
        else:
            self._headerWritten = -1

    def read(self, count=-1):
        raise RIFFChunkIOError('RIFF Write Chunk is not readable')

    def write(self, data):
        if self._closed:
            raise RIFFChunkIOError("I/O operation on closed chunk")

        if not self._headerWritten:
            self.writeHeader(False)
            chunkPos = 0
        else:
            #chunkPos = self.syncPosition()
            chunkPos = self._getChunkPosition()

        chunkSize = self.getChunkSize()
        if chunkPos < 0:
            self.seek(0)
            chunkPos = 0
        elif chunkSize is not None:
            if (chunkPos >= chunkSize):
                raise RIFFChunkIOError("Chunk is Full")
            if len(data) > (chunkSize - chunkPos):
                raise RIFFChunkIOError("Chunk does not have enough remaining space to write the data", len(data), (chunkSize - chunkPos))

        self.getHostStream().write(data)
        chunkPos += len(data)

        if (self.byteAlign is not None) and (chunkSize == chunkPos):
            alignmentFiller = '\x00' * (chunkSize % self.byteAlign)
            if alignmentFiller:
                self.getHostStream().write(alignmentFiller)
                chunkPos += len(alignmentFiller)

        self._setChunkPosition(chunkPos)
        return len(data)

    def writeAll(self, data):
        self.setChunkSize(len(data))
        if not self._headerWritten:
            self.writeHeader(True)
        self.write(data)

    def finalize(self):
        RIFFBaseChunk.finalize(self)
        if self._headerWritten != True:
            self.setChunkSize(self._getChunkPosition() or 0)
            self.writeHeader(True)
            self.seek(0,2)

    def _skipUnseekable(self):
        chunkSize = self.getChunkSize()
        if chunkSize is not None:
            chunkPos = self.tell()
            delta = max(0, chunkSize - chunkPos)
            data = min(8192, delta) * '\x00'
            while delta:
                delta -= self.write(data[:delta])


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from struct import pack, unpack
import ctypes, ctypes.util
from ctypes import cast, byref, c_void_p, c_float, POINTER

from TG.openGL.raw import aglUtils

from TG.quicktime.coreFoundationUtils import asCFString, asCFURL, c_appleid, fromAppleId, toAppleId, booleanTrue, booleanFalse
from TG.quicktime.coreVideoTexture import CVOpenGLTexture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Libraries
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

libQuickTimePath = ctypes.util.find_library("QuickTime")
libQuickTime = ctypes.cdll.LoadLibrary(libQuickTimePath)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ QuickTime Stuff
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTNewMoviePropertyElement(ctypes.Structure):
    _fields_ = [
        ('propClass', c_appleid),
        ('propID', c_appleid),
        ('propValueSize', (ctypes.c_uint32)),
        ('propValueAddress', ctypes.c_void_p),
        ('propStatus', (ctypes.c_int32)),
        ]
    
    @classmethod
    def new(klass, cid, pid, value):
        if hasattr(value, '_as_parameter_'):
            value = value._as_parameter_
        valueSize = ctypes.sizeof(type(value))
        p_value = cast(byref(value), c_void_p)
        return klass(
                fromAppleId(cid), 
                fromAppleId(pid), 
                valueSize,
                p_value, 0)

    @classmethod
    def fromProperties(klass, *properties):
        return klass.fromPropertyList(properties)

    @classmethod
    def fromPropertyList(klass, propList):
        propList = [klass.new(*p) for p in propList]
        return (klass*len(propList))(*propList)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def qtEnterMovies():
    libQuickTime.EnterMovies()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTMovieTexture(CVOpenGLTexture):
    def __init__(self, visualContext):
        CVOpenGLTexture.__init__(self)
        self.visualContext = visualContext

    def isNewImageAvailable(self):
        return libQuickTime.QTVisualContextIsNewImageAvailable(self.visualContext, None)
    def updateCVTexture(self, cvTextureRef):
        libQuickTime.QTVisualContextCopyImageForTime(self.visualContext, None, None, byref(cvTextureRef))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTOpenGLVisualContext(object):
    _as_parameter_ = None

    def __init__(self, bCreate=True):
        if bCreate:
            self.create()

    def create(self):
        if self._as_parameter_:
            return self

        cglCtx, cglPix = aglUtils.getCGLContextAndFormat()
        self._as_parameter_ = c_void_p()
        errqt = libQuickTime.QTOpenGLTextureContextCreate(None, cglCtx, cglPix, None, byref(self._as_parameter_))
        assert not errqt, errqt
        return self

    def process(self):
        libQuickTime.QTVisualContextTask(self)

    _texture = None
    def texture(self):
        tex = self._texture
        if tex is None:
            tex = QTMovieTexture(self)
            self._texture = tex
        return tex

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTMovie(object):
    def __init__(self, filePath=None):
        self.createContext()
        if filePath is not None:
            self.loadPath(filePath)

    def createContext(self):
        self.visualContext = QTOpenGLVisualContext()
        self.texMovie = self.visualContext.texture()
        
    def loadURL(self, fileURL):
        cfFileURL = asCFURL(fileURL)

        return self.loadFromProperties([
                ('dloc', 'cfur', cfFileURL),

                ('mprp', 'actv', booleanTrue),
                #('mprp', 'intn', booleanTrue),
                #('mins', 'aurn', booleanTrue),
                # No async for right now
                ('mins', 'asok', booleanTrue),

                ('ctxt', 'visu', self.visualContext),
                ])

    def loadPath(self, filePath):
        cfFilePath = asCFString(filePath)

        return self.loadFromProperties([
                ('dloc', 'cfnp', cfFilePath),

                ('mprp', 'actv', booleanTrue),
                #('mprp', 'intn', booleanTrue),
                #('mins', 'aurn', booleanTrue),
                # No async for right now
                ('mins', 'asok', booleanTrue),

                ('ctxt', 'visu', self.visualContext),
                ])

    def loadFromProperties(self, movieProperties):
        movieProperties = QTNewMoviePropertyElement.fromPropertyList(movieProperties)
        self._as_parameter_ = c_void_p()
        errqt = libQuickTime.NewMovieFromProperties(len(movieProperties), movieProperties, 0, None, byref(self._as_parameter_))

        if errqt:
            print
            print 'Movies Error:', libQuickTime.GetMoviesError()
            print 'Movie Properties::'
            for prop in movieProperties:
                print '   ', toAppleId(prop.propClass), toAppleId(prop.propID), prop.propStatus
            print
            print 
            raise Exception("Failed to initialize QuickTime movie from properties")
        elif 0:
            print 'Movie Properties::'
            for prop in movieProperties:
                print '   ', toAppleId(prop.propClass), toAppleId(prop.propID), prop.propStatus
            print
            self.printTracks()

        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def process(self, millisec=0):
        self.visualContext.process()
        return self.processMovieTask()

    def processMovieTask(self, millisec=0):
        return libQuickTime.MoviesTask(self, millisec)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def getLoadState(self):
        return libQuickTime.GetMovieLoadState(self)

    def setLooping(self, looping=1):
        libQuickTime.GoToBeginningOfMovie(self)
        timeBase = libQuickTime.GetMovieTimeBase(self)
        libQuickTime.SetTimeBaseFlags(timeBase, looping) # loopTimeBase

        hintsLoop = 0x2
        libQuickTime.SetMoviePlayHints(self, hintsLoop, hintsLoop)

    def printTracks(self):
        trackMediaType = c_appleid()
        for trackIdx in xrange(libQuickTime.GetMovieTrackCount(self)):
            track = libQuickTime.GetMovieIndTrack(self, trackIdx)
            print 'track:', trackIdx, track
            if track:
                trackMedia = libQuickTime.GetTrackMedia(track)
                print '  media:', trackMedia
                if trackMedia:
                    libQuickTime.GetMediaHandlerDescription(trackMedia, byref(trackMediaType), 0, 0)
                    print '    ', toAppleId(trackMediaType)
                    #if trackMediaType[:] not in ('vide', 'soun'):
                    #    libQuickTime.SetTrackEnabled(track, False)
                    #    print 'disabled'
                    #else:
                    #    print 'enabled'
        print


    def start(self):
        libQuickTime.StartMovie(self)

    def stop(self):
        libQuickTime.StopMovie(self)

    def isActive(self):
        return libQuickTime.GetMovieActive(self)
    def isDone(self):
        return libQuickTime.IsMovieDone(self)

    def ptInMovie(self):
        return libQuickTime.PtInMovie(self)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _availFrame = 0
    def _onImageAvailable(self, visualContext, syncTimeStamp, userParam):
        self._availFrame += 1

    def _setupImageAvailableCB(self):
        QTVisualContextImageAvailableCallback = ctypes.CFUNCTYPE(None, c_void_p, c_uint32, c_void_p)
        onImageAvailableCallback = QTVisualContextImageAvailableCallback(self._onImageAvailable)
        libQuickTime.QTVisualContextSetImageAvailableCallback(self.visualContext, onImageAvailableCallback, None)


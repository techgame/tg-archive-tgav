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

class TimeRecord(ctypes.Structure):
    _fields_ = [
        ('value', ctypes.c_long),
        ('scale', ctypes.c_long),
        ('base', ctypes.c_void_p),
        ]

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
    def fromPropertyList(klass, *propLists):
        propList = [klass.new(*p) for propList in propLists for p in propList]
        return (klass*len(propList))(*propList)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def qtEnterMovies():
    libQuickTime.EnterMovies()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTTexture(CVOpenGLTexture):
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

    _qtTexture = None
    def qtTexture(self):
        tex = self._qtTexture
        if tex is None:
            tex = QTTexture(self)
            self._qtTexture = tex
        return tex

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QTMovie(object):
    def __init__(self, path=None):
        self.createContext()
        if path is not None:
            self.loadPath(path)

    def loadPath(self, path):
        if '://' in path:
            return self.loadURL(path)
        else:
            return self.loadFilePath(path)

    def loadURL(self, urlPath):
        cfFileURL = asCFURL(urlPath)

        return self.loadFromProperties([
                ('dloc', 'cfur', cfFileURL),
                ('ctxt', 'visu', self.visualContext),
                ])

    def loadFilePath(self, filePath):
        cfFilePath = asCFString(filePath)

        return self.loadFromProperties([
                ('dloc', 'cfnp', cfFilePath),
                ('ctxt', 'visu', self.visualContext),
                ])

    def loadFromProperties(self, movieProperties, 
            defaultMovieProperties=[
                ('mprp', 'actv', booleanTrue), # set movie active after loading
                ('mprp', 'intn', booleanTrue), # don't interact with user
                ('mins', 'aurn', booleanTrue), # don't ask user help for unresolved references
                ('mins', 'asok', booleanTrue), # load asynchronously
                ]):
        movieProperties = QTNewMoviePropertyElement.fromPropertyList(movieProperties, defaultMovieProperties)
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

    def createContext(self):
        self.visualContext = QTOpenGLVisualContext()
        self.qtTexture = self.visualContext.qtTexture()
        
    def process(self, seconds=0):
        r = self.visualContext.process()
        self.processMovieTask(seconds)
        return r

    def processMovieTask(self, seconds=0):
        return libQuickTime.MoviesTask(self, int(seconds*1000))

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


    def getMovieRate(self):
        return libQuickTime.GetMovieRate(self)
    def getMovieTime(self):
        # Sending a value of None to this method will only return the current time
        # value, versus the time value and the pointer to the time structure
        return libQuickTime.GetMovieTime(self, None)


    def setMovieTime(self, pos):
        timeRecord = TimeRecord()
        libQuickTime.GetMovieTime(self, byref(timeRecord))
        timeRecord.value = pos
        return libQuickTime.SetMovieTime(self, byref(timeRecord))



    def getMovieDuration(self):
        return libQuickTime.GetMovieDuration(self)
    def getMovieTimeScale(self):
        return libQuickTime.GetMovieTimeScale(self)

    def start(self):
        libQuickTime.StartMovie(self)

    def stop(self):
        libQuickTime.StopMovie(self)

    def pause(self):
        libQuickTime.SetMovieRate(self, 0)

    def goToBeginning(self):
        libQuickTime.GoToBeginningOfMovie(self)


    def isActive(self):
        return libQuickTime.GetMovieActive(self)
    def isDone(self):
        return libQuickTime.IsMovieDone(self)

    def ptInMovie(self):
        return libQuickTime.PtInMovie(self)


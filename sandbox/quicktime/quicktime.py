
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ AGL Stuff
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    AGL_SWAP_RECT = 200
    AGL_BUFFER_RECT = 202
    AGL_SWAP_LIMIT = 203
    AGL_COLORMAP_TRACKING = 210
    AGL_COLORMAP_ENTRY = 212
    AGL_RASTERIZATION = 220
    AGL_SWAP_INTERVAL = 222
    AGL_STATE_VALIDATION = 230
    AGL_BUFFER_NAME = 231
    AGL_ORDER_CONTEXT_TO_FRONT = 232
    AGL_CONTEXT_SURFACE_ID = 233
    AGL_CONTEXT_DISPLAY_ID = 234
    AGL_SURFACE_ORDER = 235
    AGL_SURFACE_OPACITY = 236
    AGL_CLIP_REGION = 254
    AGL_FS_CAPTURE_SINGLE = 255
    AGL_SURFACE_BACKING_SIZE = 304
    AGL_ENABLE_SURFACE_BACKING_SIZE = 305
    AGL_SURFACE_VOLATILE = 306

    AGL_NONE = 0
    AGL_ALL_RENDERERS = 1
    AGL_BUFFER_SIZE = 2
    AGL_LEVEL = 3
    AGL_RGBA = 4
    AGL_DOUBLEBUFFER = 5
    AGL_STEREO = 6
    AGL_AUX_BUFFERS = 7
    AGL_RED_SIZE = 8
    AGL_GREEN_SIZE = 9
    AGL_BLUE_SIZE = 10
    AGL_ALPHA_SIZE = 11
    AGL_DEPTH_SIZE = 12
    AGL_STENCIL_SIZE = 13
    AGL_ACCUM_RED_SIZE = 14
    AGL_ACCUM_GREEN_SIZE = 15
    AGL_ACCUM_BLUE_SIZE = 16
    AGL_ACCUM_ALPHA_SIZE = 17
    AGL_PIXEL_SIZE = 50
    AGL_MINIMUM_POLICY = 51
    AGL_MAXIMUM_POLICY = 52
    AGL_OFFSCREEN = 53
    AGL_FULLSCREEN = 54
    AGL_SAMPLE_BUFFERS_ARB = 55
    AGL_SAMPLES_ARB = 56
    AGL_AUX_DEPTH_STENCIL = 57
    AGL_COLOR_FLOAT = 58
    AGL_MULTISAMPLE = 59
    AGL_SUPERSAMPLE = 60
    AGL_SAMPLE_ALPHA = 61

    libAGL = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AGL"))
    def getAGLError():
        err = libAGL.aglGetError()
        errStr = cast(libAGL.aglErrorString(err), ctypes.c_char_p)
        return errStr.value, err

    aglCtx = libAGL.aglGetCurrentContext()
    libAGL.aglSetInteger(aglCtx, AGL_SWAP_INTERVAL, byref(GLint(1)))

    # getting cglContext and cglPixel format to initialize the move from
    cglCtx = c_void_p()
    libAGL.aglGetCGLContext(aglCtx, byref(cglCtx))

    aglAttribs = [
        AGL_RGBA, 
        #AGL_DOUBLEBUFFER, 
        AGL_MINIMUM_POLICY, 
        #AGL_DEPTH_SIZE, 1,
        AGL_RED_SIZE, 1, 
        AGL_GREEN_SIZE, 1, 
        AGL_BLUE_SIZE, 1, 
        AGL_ALPHA_SIZE, 1, 
        AGL_NONE]

    aglAttribs = asArray(aglAttribs, GLint)
    aglPix = libAGL.aglChoosePixelFormat(None, 0, aglAttribs)

    cglPix = c_void_p()
    libAGL.aglGetCGLPixelFormat(aglPix, byref(cglPix))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ QuickTime Stuff
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    char4 = ctypes.c_char * 4

    libCF = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    libQT = ctypes.cdll.LoadLibrary(ctypes.util.find_library("QuickTime"))
    libCV = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreVideo"))

    libQT.EnterMovies()

    qtVisualContext = c_void_p()

    class QTNewMoviePropertyElement(ctypes.Structure):
        _fields_ = [
            ('propClass', (char4)),
            ('propID', (char4)),
            ('propValueSize', (ctypes.c_uint32)),
            ('propValueAddress', ctypes.c_void_p),
            ('propStatus', (ctypes.c_int32)),
            ]

    def vpByref(i):
        return cast(byref(i), c_void_p)

    Boolean = ctypes.c_ubyte
    CFStringRef = ctypes.c_void_p

    channelMovie = c_void_p()

    filePath = os.path.abspath("Curious George.m4b")
    #filePath = os.path.abspath("cercle.mov")
    #filePath = "/Users/shane/Movies/Testing.mov"
    #filePath = "/Users/shane/Movies/UnionStation_02.wmv"
    #filePath = "/Users/shane/Movies/AspyrIntro.mov"
    #filePath = "/Users/shane/Movies/jumps.mov"
    cfFilePath = CFStringRef(libCF.CFStringCreateWithCString(0, ctypes.c_char_p(filePath.encode('utf8')), 0x8000100)) # kCFStringEncodingUTF8 = 0x8000100
    assert len(filePath) == libCF.CFStringGetLength(cfFilePath)
    assert os.path.isfile(filePath)

    bFalse = Boolean(0)
    bTrue = Boolean(1)
    def qtMovieProp(cid, pid, vSize, addr):
        return QTNewMoviePropertyElement(cid[::-1], pid[::-1], ctypes.sizeof(vSize), vpByref(addr), 0)
    newMovieProperties = asArray([
        qtMovieProp('dloc', 'cfnp', cfFilePath, cfFilePath),

        qtMovieProp('mprp', 'actv', Boolean, bTrue),
        qtMovieProp('mprp', 'intn', Boolean, bTrue),

        qtMovieProp('mins', 'aurn', Boolean, bTrue),

        # No async for right now
        qtMovieProp('mins', 'asok', Boolean, bTrue),

        qtMovieProp('ctxt', 'visu', qtVisualContext, qtVisualContext),
        ], QTNewMoviePropertyElement)

    theError = libQT.QTOpenGLTextureContextCreate(None, cglCtx, cglPix, None, byref(qtVisualContext))
    print 'QT Texture:', theError, qtVisualContext, bool(qtVisualContext)

    theError = libQT.NewMovieFromProperties(len(newMovieProperties), newMovieProperties, 0, None, byref(channelMovie))
    print 'QT NewMovie:', theError, channelMovie, bool(channelMovie)
    for prop in newMovieProperties:
        print '   ', prop.propClass, prop.propID, prop.propStatus
    print
    if theError:
        print 'Movies Error:', libQT.GetMoviesError()
        print "Quicktime failed to initialize, bailing"
        sys.exit(1)
    loadState = libQT.GetMovieLoadState(channelMovie)
    print 'Movie Load State:', hex(loadState)

    MoviesTask = libQT.MoviesTask
    MoviesTask(channelMovie, 0)

    libQT.GoToBeginningOfMovie(channelMovie)
    timeBase = libQT.GetMovieTimeBase(channelMovie)
    libQT.SetTimeBaseFlags(timeBase, 0x1) # loopTimeBase
    hintsLoop = 0x2
    libQT.SetMoviePlayHints(channelMovie, hintsLoop, hintsLoop)

    trackMediaType = char4()
    for trackIdx in xrange(libQT.GetMovieTrackCount(channelMovie)):
        track = libQT.GetMovieIndTrack(channelMovie, trackIdx)
        print 'track:', trackIdx, track
        if track:
            trackMedia = libQT.GetTrackMedia(track)
            print '  media:', trackMedia
            if trackMedia:
                libQT.GetMediaHandlerDescription(trackMedia, byref(trackMediaType), 0, 0)
                print '    ', trackMediaType[:], 
                #if trackMediaType[:] not in ('vide', 'soun'):
                #    libQT.SetTrackEnabled(track, False)
                #    print 'disabled'
                #else:
                #    print 'enabled'
    print


    MoviesTask(channelMovie, 0)
    libQT.StartMovie(channelMovie)

    QTVisualContextTask = libQT.QTVisualContextTask
    QTVisualContextSetImageAvailableCallback = libQT.QTVisualContextSetImageAvailableCallback
    QTVisualContextIsNewImageAvailable = libQT.QTVisualContextIsNewImageAvailable
    QTVisualContextCopyImageForTime = libQT.QTVisualContextCopyImageForTime

    CVOpenGLTextureRetain = libCV.CVOpenGLTextureRetain
    CVOpenGLTextureRelease = libCV.CVOpenGLTextureRelease
    CVOpenGLTextureGetCleanTexCoords = libCV.CVOpenGLTextureGetCleanTexCoords
    CVOpenGLTextureGetTarget = libCV.CVOpenGLTextureGetTarget
    CVOpenGLTextureGetName = libCV.CVOpenGLTextureGetName 

    CVOpenGLTextureRef = c_void_p

    currentTextureRef = CVOpenGLTextureRef()
    GLfloat2 = GLfloat*2
    ll = GLfloat2(); lr = GLfloat2(); ul = GLfloat2(); ur = GLfloat2();

    QTVisualContextImageAvailableCallback = ctypes.CFUNCTYPE(None, c_void_p, c_uint32, c_void_p)
    #def onImageAvailable(qtVisualContext, syncTimeStamp, userParam):
    #    print 'ia:', syncTimeStamp
    #    global currentTextureRef, ll, lr, ur, ul
    #    try:
    #        oldTextureRef = currentTextureRef

    #        newTextureRef = CVOpenGLTextureRef()
    #        QTVisualContextCopyImageForTime(qtVisualContext, None, syncTimeStamp, byref(newTextureRef))

    #        CVOpenGLTextureGetCleanTexCoords(newTextureRef, ll, lr, ur, ul)

    #        renderLock.acquire()
    #        try:
    #            currentTextureRef = newTextureRef
    #        finally:
    #            renderLock.release()
    #        CVOpenGLTextureRelease(oldTextureRef)

    #    except NameError:
    #        QTVisualContextSetImageAvailableCallback(qtVisualContext, None, None)
    #        raise

    #onImageAvailableCallback = QTVisualContextImageAvailableCallback(onImageAvailable)
    #QTVisualContextSetImageAvailableCallback(qtVisualContext, onImageAvailableCallback, None)

    def onImageAvailable(qtVisualContext, syncTimeStamp, userParam):
        global onImgAvailableCount
        onImgAvailableCount += 1
        #print 'swh:', onImgAvailableCount
    onImageAvailableCallback = QTVisualContextImageAvailableCallback(onImageAvailable)
    QTVisualContextSetImageAvailableCallback(qtVisualContext, onImageAvailableCallback, None)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#!/usr/local/bin/python2.5
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~ Copyright (C) 2002-2004  TechGame Networks, LLC.
##~ 
##~ This library is free software; you can redistribute it and/or
##~ modify it under the terms of the BSD style License as found in the 
##~ LICENSE file included with this distribution.
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from TG.skinning.toolkits.wx import wxSkinModel, XMLSkin

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xmlSkin = XMLSkin("""<?xml version='1.0'?>
<skin xmlns='TG.skinning.toolkits.wx'>
    <style>
        frame {frame-main:1; locking:1; size:'1024,768'}
        frame>layout {layout-cfg:'1,EXPAND'}
        frame>layout>panel {layout-cfg:'1,EXPAND'}
    </style>

    <frame title='GLCanvas in a Skin' show='1'>
        <layout>
            <opengl-canvas ctxobj='canvas' layout-cfg='1, EXPAND' gl-style='WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 8'>
                import sys
                import time
                timestamp = time.time # time.clock

                import threading
                renderLock = threading.Lock()

                onImgAvailableCount = 0

                frame = ctx.frame
                title = frame.GetTitle()
                obj.winframeLog = []
                obj.glframeLog = []
                obj.lastUpdate = timestamp()

                obj.OnEraseBackground = lambda  s, *args: None

                def updateFPS(obj, winstart, glstart, glend, winend, winframeLog=obj.winframeLog, glframeLog=obj.glframeLog):
                    winframeLog.append(winend-winstart)
                    glframeLog.append(glend-glstart)

                    delta = winend - obj.lastUpdate
                    if delta >= 1:
                        obj.lastUpdate = winend
                        count = len(winframeLog)

                        winTime = sum(winframeLog, 0.0)
                        if winTime:
                            winfps = count / winTime
                        else: winfps = -1
                        winframeLog[:] = winframeLog[-10:]

                        glTime = sum(glframeLog, 0.0)
                        if glTime:
                            glfps = count / sum(glframeLog, 0.0)
                        else: glfps = -1
                        glframeLog[:] = glframeLog[-10:]

                        global onImgAvailableCount
                        fpsStr = '%.6s : %.6s FPS ratio: %.4s (effective), %.6s frames, %s movie frames' % (glfps, winfps, glfps/winfps, count/delta, onImgAvailableCount)
                        frame.SetTitle('%s - %s' % (title, fpsStr))
                        print '\\r', fpsStr.ljust(75),
                        sys.stdout.flush()

                        onImgAvailableCount = 0
                        global loadState
                        newLoadState = libQT.GetMovieLoadState(channelMovie)
                        if newLoadState != loadState:
                            loadState = newLoadState
                            print 'Movie Load State:', hex(loadState)


                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                import os, sys
                import array
                import ctypes, ctypes.util
                from ctypes import cast, byref, c_void_p, c_float, POINTER
                from TG.openGL.raw.gl import *
                from TG.openGL.raw.glu import *
                from TG.openGL.raw.glext import GL_TEXTURE_RECTANGLE_ARB

                def asArray(lst, elementType=None):
                    elementType = elementType or lst[0].__class__
                    arrayType = elementType * len(lst)
                    return arrayType(*lst)

                GLfloat_p = POINTER(GLfloat)
                GLfloat4 = (GLfloat*4)
                GLfloat16 = (GLfloat*16)

                obj.SetCurrent()

                glEnable(GL_DEPTH_TEST)
                glEnable(GL_COLOR_MATERIAL)
                #glPolygonMode(GL_FRONT, GL_LINE)
                #glPolygonMode(GL_BACK, GL_FILL)
                #glCullFace(GL_BACK)
                #glEnable(GL_CULL_FACE)

                glShadeModel(GL_SMOOTH)
                glEnable(GL_LIGHTING)
                glEnable(GL_LIGHT0)

                iSize = 2**8; jSize = iSize

                vstripes = array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                                    for c in (255*(1 &amp; (j&gt;&gt;2)),)
                                                        for e in (0,0,c,c)])
                vstripesPtr = c_void_p(vstripes.buffer_info()[0])
                vstripesTexName = GLenum(0)
                glGenTextures(1, byref(vstripesTexName))

                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

                glActiveTexture(GL_TEXTURE0)
                for texType, texName, texPtr in [(GL_TEXTURE_2D, vstripesTexName, vstripesPtr)]:
                    glBindTexture(texType, texName)

                    glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                    glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameterfv(texType, GL_TEXTURE_BORDER_COLOR, GLfloat4(0., 0., 0., 0.))

                    glTexParameteri(texType, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                    glTexParameteri(texType, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

                    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_COMPRESSED_RGBA, iSize, jSize, GL_RGBA, GL_UNSIGNED_BYTE, texPtr)

                glEnable(GL_TEXTURE_2D)
                glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

                if 0:
                    glMatrixMode (GL_TEXTURE)
                    glLoadIdentity ()

                    # viewport transform
                    glTranslatef(0.5,0.5,0.0)
                    glScalef(0.5,0.5,1.0)

                    # projection transform
                    #gluPerspective(15, 1., 1., 10.)
                    #gluLookAt(0,0,15,  0,0,0,  0,1,0)

                    matrixProjector = GLfloat16()
                    glGetFloatv(GL_TRANSPOSE_MODELVIEW_MATRIX, matrixProjector)

                    glMatrixMode (GL_MODELVIEW)

                    for axis, offset in [(GL_S, 0), (GL_T, 4), (GL_R, 8), (GL_Q, 12)]:
                        glTexGeni(axis, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR)
                        glTexGenfv(axis, GL_EYE_PLANE, cast(byref_at(matrixProjector, offset), CLfloat_p))

                    map(glEnable, (GL_TEXTURE_GEN_S, GL_TEXTURE_GEN_T, GL_TEXTURE_GEN_R, GL_TEXTURE_GEN_Q))

                glActiveTexture(GL_TEXTURE0)

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                quad = gluNewQuadric()
                gluQuadricDrawStyle(quad, GLU_FILL)
                #gluQuadricDrawStyle(quad, GLU_LINE)
                #gluQuadricDrawStyle(quad, GLU_POINT)
                gluQuadricNormals(quad, GLU_SMOOTH)
                gluQuadricTexture(quad, True)

                sphereName = glGenLists(1)
                glNewList(sphereName, GL_COMPILE)
                glBindTexture(GL_TEXTURE_2D, vstripesTexName)

                slices = stacks = 50
                gluSphere (quad, 5.0, slices, stacks)
                #gluSphere (quad, 4.0, slices, stacks)
                #gluSphere (quad, 3.0, slices, stacks)
                #gluSphere (quad, 2.0, slices, stacks)
                #gluSphere (quad, 1.0, slices, stacks)
                glEndList()
                glClearColor(0.0, 0.0, 0.0, 0.0)

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

                @ctx.add
                def resize(obj=obj):
                    (w,h) = obj.GetSize()
                    if not w or not h: return

                    #obj.SetCurrent()
                    glViewport (0, 0, w, h)
                    glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
                    glMatrixMode (GL_PROJECTION)
                    glLoadIdentity ()
                    gluPerspective(45, float(w)/h, 1, 100)
                    glMatrixMode (GL_MODELVIEW)
                    glLoadIdentity ()
                    gluLookAt (0,0,15,   0,0,0,   0,1,0)

                @ctx.add
                def render(obj=obj, timestamp=0):
                    global currentTextureRef, ll, lr, ur, ul
                    c = timestamp*5*2
                    if renderLock.acquire(0):
                        try:
                            glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

                            glPushMatrix()

                            glRotatef((c*2) % 360.0, 0, 0, 1)
                            glRotatef((c*3) % 360.0, 0, 1, 0)
                            glRotatef((c*5) % 360.0, 1, 0, 0)

                            #glEnable(GL_TEXTURE_2D)
                            if 0:
                                glBindTexture(GL_TEXTURE_2D, vstripesTexName)
                                glCallList(sphereName)

                            #glPopMatrix()
                            #glPushMatrix()
                            if QTVisualContextIsNewImageAvailable(qtVisualContext, None):
                                #print 'NewImgAvail'
                                CVOpenGLTextureRelease(currentTextureRef)
                                currentTextureRef = CVOpenGLTextureRef()
                                QTVisualContextCopyImageForTime(qtVisualContext, None, None, byref(currentTextureRef))
                                CVOpenGLTextureGetCleanTexCoords(currentTextureRef, ll, lr, ur, ul)

                            if currentTextureRef:

                                textureTarget = CVOpenGLTextureGetTarget(currentTextureRef)
                                textureName = CVOpenGLTextureGetName(currentTextureRef)

                                glEnable(textureTarget)
                                glBindTexture(textureTarget, textureName)

                                glScalef(4,4,1)
                                glColor4f(1., 1., 1., 0.8)

                                glBegin(GL_QUADS)

                                #print ul[:], ll[:], lr[:], ur[:]

                                glTexCoord2fv(ul)
                                #glTexCoord2f(-1., 1.)
                                glVertex2f(-1.333, 1.)

                                glTexCoord2fv(ll)
                                #glTexCoord2f(-1., -1.)
                                glVertex2f(-1.333, -1.)

                                glTexCoord2fv(lr)
                                #glTexCoord2f(1., -1.)
                                glVertex2f(1.333, -1.)

                                glTexCoord2fv(ur)
                                #glTexCoord2f(1., 1.)
                                glVertex2f(1.333, 1.)

                                glEnd()


                            glPopMatrix()

                            glDisable(GL_TEXTURE_RECTANGLE_ARB)
                            glDisable(GL_TEXTURE_2D)
                            glFinish ()
                        finally:
                            renderLock.release()

                @ctx.add
                def refresh(obj=obj):
                    QTVisualContextTask(qtVisualContext)
                    MoviesTask(channelMovie, 0)

                    glstart = timestamp()
                    winstart = glstart
                    render(obj, glstart)
                    glend = timestamp()
                    
                    obj.SwapBuffers()
                    winend = timestamp()

                    updateFPS(obj, winstart, glstart, glend, winend)

                <event type="EVT_ERASE_BACKGROUND"/>
                <event>
                    PaintDC(obj)
                    refresh()

                    # uncomment the following to enter an update loop
                    #wx.FutureCall(10, obj.Refresh)
                </event>
                <event type="EVT_SIZE">
                    resize()
                    refresh()
                </event>                        

                <timer seconds='1/50.'>
                    @ctx.add
                    def toggle():
                        if obj.IsRunning():
                            obj.Stop()
                        else: obj.Start()

                    <event>
                        ctx.refresh()
                        #ctx.canvas.Refresh()
                    </event>
                </timer>
            </opengl-canvas>
        </layout>
    </frame>
</skin>
""")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    wxSkinModel.fromSkin(xmlSkin).skinModel()



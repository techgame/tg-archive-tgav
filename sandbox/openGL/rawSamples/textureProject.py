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
            <panel>
                <layout>
                    <opengl-canvas ctxobj='canvas' layout-cfg='1, EXPAND' gl-style='WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 8'>
                        import sys
                        import time
                        timestamp = time.time # time.clock

                        frame = ctx.frame
                        title = frame.GetTitle()
                        obj.winframeLog = []
                        obj.glframeLog = []
                        obj.lastUpdate = timestamp()

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

                                fpsStr = '%.6s : %.6s FPS ratio: %.4s (effective), %.6s frames' % (glfps, winfps, glfps/winfps, count/delta)
                                frame.SetTitle('%s - %s' % (title, fpsStr))
                                print '\\r', fpsStr.ljust(75),
                                sys.stdout.flush()

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        import array
                        from ctypes import cast, byref, byref_at, c_void_p, c_float, POINTER
                        c_float_p = POINTER(c_float)
                        from TG.openGL.raw import gl, glu

                        obj.SetCurrent()
                        quad = glu.gluNewQuadric()
                        glu.gluQuadricDrawStyle(quad, glu.GLU_FILL)
                        #glu.gluQuadricDrawStyle(quad, glu.GLU_LINE)
                        #glu.gluQuadricDrawStyle(quad, glu.GLU_POINT)
                        glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
                        glu.gluQuadricTexture(quad, True)
                        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

                        gl.glEnable(gl.GL_DEPTH_TEST)
                        #gl.glEnable(gl.GL_COLOR_MATERIAL)
                        #gl.glPolygonMode(gl.GL_FRONT, gl.GL_LINE)
                        #gl.glPolygonMode(gl.GL_BACK, gl.GL_FILL)
                        gl.glCullFace(gl.GL_BACK)
                        gl.glEnable(gl.GL_CULL_FACE)

                        gl.glShadeModel(gl.GL_SMOOTH)
                        gl.glEnable(gl.GL_LIGHTING)
                        gl.glEnable(gl.GL_LIGHT0)

                        iSize = 2**8; jSize = iSize

                        checkerBoard = array.array('B', (iSize*jSize*4)*[0])
                        for i in xrange(0, iSize):
                            rowOff = i*jSize
                            for j in xrange(0, jSize):
                                off = (rowOff+j)*4
                                c = (i*j) &amp; 0xff
                                if i > j:
                                    checkerBoard[off+0] = 64
                                else:
                                    checkerBoard[off+1] = 64
                                checkerBoard[off+2] = (i*j) &amp; 0xff
                                #checkerBoard[off+2] = 0
                                checkerBoard[off+3] = 225

                        checkerBoardPtr = c_void_p(checkerBoard.buffer_info()[0])
                        checkerboardTexName = gl.GLenum(0)
                        gl.glGenTextures(1, byref(checkerboardTexName))

                        vstripes = array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                                            for c in (255*(1 &amp; (j&gt;&gt;3)),)
                                                                for e in (0,0,c,c)])
                        vstripesPtr = c_void_p(vstripes.buffer_info()[0])
                        vstripesTexName = gl.GLenum(0)
                        gl.glGenTextures(1, byref(vstripesTexName))

                        hstripes = array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                                            for c in (255*(1 &amp; (i&gt;&gt;3)),)
                                                                for e in (c,0,0,c)])
                        hstripesPtr = c_void_p(hstripes.buffer_info()[0])
                        hstripesTexName = gl.GLenum(0)
                        gl.glGenTextures(1, byref(hstripesTexName))


                        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

                        gl.glActiveTexture(gl.GL_TEXTURE0)

                        for texName, texPtr in [
                                (vstripesTexName, vstripesPtr),
                                (hstripesTexName, hstripesPtr),
                                ]:
                            gl.glBindTexture(gl.GL_TEXTURE_2D, texName)

                            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)

                            glu.gluBuild2DMipmaps(gl.GL_TEXTURE_2D, gl.GL_RGBA, iSize, jSize, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texPtr)

                        gl.glEnable(gl.GL_TEXTURE_2D)
                        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)

                        gl.glMatrixMode (gl.GL_TEXTURE)
                        gl.glLoadIdentity ()
                        gl.glMatrixMode (gl.GL_MODELVIEW)

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        gl.glActiveTexture(gl.GL_TEXTURE1)

                        gl.glBindTexture(gl.GL_TEXTURE_2D, checkerboardTexName)

                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                        borderColor = (gl.GLfloat*4)()
                        for i, e in enumerate([0.0, 0.0, 0.0, 0.0]): 
                            borderColor[i] = e
                        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, borderColor);

                        glu.gluBuild2DMipmaps(gl.GL_TEXTURE_2D, gl.GL_RGBA, iSize, jSize, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, checkerBoardPtr)

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        gl.glEnable(gl.GL_TEXTURE_2D)
                        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)

                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        gl.glMatrixMode (gl.GL_TEXTURE)
                        gl.glLoadIdentity ()

                        # viewport transform
                        gl.glTranslatef(0.5,0.5,0.0)
                        gl.glScalef(0.5,0.5,1.0)

                        # projection transform
                        glu.gluPerspective(15, 1., 1., 10.)
                        glu.gluLookAt(0,0,15,  0,0,0,  0,1,0)

                        matrixProjector = (c_float*16)()
                        gl.glGetFloatv(gl.GL_TRANSPOSE_MODELVIEW_MATRIX, matrixProjector)

                        gl.glMatrixMode (gl.GL_MODELVIEW)

                        for axis, offset in [(gl.GL_S, 0), (gl.GL_T, 4), (gl.GL_R, 8), (gl.GL_Q, 12)]:
                            if 0:
                                gl.glTexGeni(axis, gl.GL_TEXTURE_GEN_MODE, gl.GL_OBJECT_LINEAR);
                                gl.glTexGenfv(axis, gl.GL_OBJECT_PLANE, cast(byref_at(matrixProjector, offset), c_float_p))
                            else:
                                gl.glTexGeni(axis, gl.GL_TEXTURE_GEN_MODE, gl.GL_EYE_LINEAR);
                                gl.glTexGenfv(axis, gl.GL_EYE_PLANE, cast(byref_at(matrixProjector, offset), c_float_p))

                        map(gl.glEnable, (gl.GL_TEXTURE_GEN_S, gl.GL_TEXTURE_GEN_T, gl.GL_TEXTURE_GEN_R, gl.GL_TEXTURE_GEN_Q))

                        gl.glActiveTexture(gl.GL_TEXTURE0)
                        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                        @ctx.add
                        def resize(obj=obj):
                            (w,h) = obj.GetSize()
                            if not w or not h: return

                            #obj.SetCurrent()
                            gl.glViewport (0, 0, w, h)
                            gl.glMatrixMode (gl.GL_PROJECTION)
                            gl.glLoadIdentity ()
                            glu.gluPerspective(45, float(w)/h, 1, 100)
                            gl.glMatrixMode (gl.GL_MODELVIEW)
                            gl.glLoadIdentity ()
                            glu.gluLookAt (0,0,15,   0,0,0,   0,1,0)

                        @ctx.add
                        def render(obj=obj, timestamp=0):
                            gl.glClear (gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)

                            #gl.glEnable(gl.GL_TEXTURE_2D)
                            c = timestamp*5

                            if 0:
                                gl.glMatrixMode (gl.GL_TEXTURE)

                                gl.glActiveTexture(gl.GL_TEXTURE1)
                                gl.glPushMatrix()
                                m = abs(1. - (timestamp/2 % 2.0))
                                gl.glRotatef(10*(1.-2*m), 1, 0, 0)

                                m = abs(1. - (timestamp/3 % 2.0))
                                gl.glRotatef(10*(1.-2*m), 0, 1, 0)
                                gl.glActiveTexture(gl.GL_TEXTURE0)

                                gl.glMatrixMode (gl.GL_MODELVIEW)
                            gl.glPushMatrix()
                            gl.glRotatef((c*4) % 360.0, 0, 0, 1)
                            gl.glRotatef((c*3) % 360.0, 0, 1, 0)
                            gl.glRotatef((c*5) % 360.0, 1, 0, 0)

                            #gl.glBindTexture(gl.GL_TEXTURE_2D, vstripesTexName)
                            #glu.gluSphere (quad, 1.0, 10, 10)

                            gl.glPushMatrix()
                            gl.glTranslatef(0.,-1.,4.)
                            gl.glColor3f (1.0, 0.0, 1.0)
                            #gl.glRotatef((c*-12) % 360.0, 1, 0, 1)
                            gl.glBindTexture(gl.GL_TEXTURE_2D, hstripesTexName)
                            glu.gluSphere (quad, 4.0, 40, 40)

                            gl.glPopMatrix()
                            gl.glPushMatrix()
                            gl.glTranslatef(-2.,4.,0.)
                            #gl.glRotatef((c*-8) % 360.0, 1, 1, 0)

                            gl.glColor3f (0.0, 1.0, 1.0)
                            gl.glBindTexture(gl.GL_TEXTURE_2D, hstripesTexName)
                            glu.gluSphere (quad, 3.0, 40, 40)

                            gl.glPopMatrix()
                            gl.glPushMatrix()
                            gl.glTranslatef(4.,0.,-2.)
                            #gl.glRotatef((c*-10) % 360.0, 0, 1, 1)
                            gl.glColor3f (1.0, 1.0, 0.0)
                            gl.glBindTexture(gl.GL_TEXTURE_2D, vstripesTexName)
                            glu.gluSphere (quad, 3.5, 40, 40)

                            gl.glPopMatrix()

                            gl.glPopMatrix()

                            if 0:
                                gl.glMatrixMode (gl.GL_TEXTURE)

                                gl.glActiveTexture(gl.GL_TEXTURE1)
                                gl.glPopMatrix()
                                gl.glActiveTexture(gl.GL_TEXTURE0)

                                gl.glMatrixMode (gl.GL_MODELVIEW)

                            #gl.glDisable(gl.GL_TEXTURE_2D)

                            #gl.glFlush ()
                            gl.glFinish ()

                        @ctx.add
                        def refresh(obj=obj):
                            #obj.SetCurrent()

                            glstart = timestamp()
                            winstart = glstart
                            render(obj, glstart)
                            glend = timestamp()
                            
                            obj.SwapBuffers()
                            winend = timestamp()

                            updateFPS(obj, winstart, glstart, glend, winend)

                        <event type="EVT_ERASE_BACKGROUND"/>
                        <event>
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
                    <!--layout layout-cfg='0, EXPAND'>
                        <button text='Enabled'><event>ctx.toggle()</event></button>
                        <pycrust layout-minsize='100,300' layout-cfg='1,EXPAND'/>
                    </layout-->
                </layout>
            </panel>
        </layout>
    </frame>
</skin>
""")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    wxSkinModel.fromSkin(xmlSkin).skinModel()



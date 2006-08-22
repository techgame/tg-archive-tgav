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
        <menubar>
            <menu text='View'>
                <item text='Full Screen\tCtrl-F' help='Shows My Frame on the entire screen'>
                    display = None
                    def getDisplay():
                        global display
                        if display is None:
                            display = wx.Display(wx.Display.GetFromPoint(ctx.frame.GetPosition()))
                        return display
                    def delDisplay():
                        global display
                        display = None

                    oldMode = None
                    def changeMode(*args):
                        global oldMode
                        if oldMode is None:
                            oldMode = getDisplay().GetCurrentMode()
                        return getDisplay().ChangeMode(wx.VideoMode(*args))

                    def restoreMode():
                        global oldMode
                        if oldMode is not None:
                            getDisplay().ChangeMode(oldMode)
                            oldMode = None
                        delDisplay()

                    <event>

                        if ctx.frame.IsFullScreen():
                            print 'reset mode:', restoreMode()
                            ctx.frame.ShowFullScreen(False)
                        else:
                            print 'change mode:', changeMode(1920,1200,32)
                            ctx.frame.ShowFullScreen(True)
                    </event>
                    <event type='EVT_UPDATE_UI'>
                        if ctx.frame.IsFullScreen():
                            obj.SetText('Restore from Full Screen\tCtrl-F')
                        else:
                            obj.SetText('Full Screen\tCtrl-F')
                    </event>
                </item>
            </menu>
        </menubar>
        <layout>
            <panel>
                <layout>
                    <opengl-canvas ctxobj='canvas' layout-cfg='1, EXPAND'
                            gl-style='WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 8'>
                        import sys
                        import time
                        #timestamp = timestamp
                        timestamp = time.time

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
                        gl.glEnable(gl.GL_COLOR_MATERIAL)

                        gl.glShadeModel(gl.GL_SMOOTH)
                        gl.glEnable(gl.GL_LIGHTING)
                        gl.glEnable(gl.GL_LIGHT0)
                        #gl.glPolygonMode(gl.PASS)

                        @ctx.add
                        def render(obj=obj, timestamp=0):
                            gl.glClear (gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)
                            gl.glPushMatrix()

                            gl.glTranslatef(0.,0.,25.)
                            c = timestamp*20
                            gl.glRotatef((c*4) % 360.0, 0, 0, 1)
                            gl.glRotatef((c*3) % 360.0, 0, 1, 0)
                            gl.glRotatef((c*5) % 360.0, 1, 0, 0)

                            gl.glPushMatrix()
                            gl.glTranslatef(0.,0.,5.)
                            gl.glColor3f (1.0, 0.0, 1.0)
                            glu.gluSphere (quad, 2.0, 20, 20)

                            gl.glPopMatrix()
                            gl.glPushMatrix()
                            gl.glTranslatef(0.,5.,0.)

                            gl.glColor3f (0.0, 1.0, 1.0)
                            glu.gluSphere (quad, 3.0, 30, 30)

                            gl.glPopMatrix()
                            gl.glPushMatrix()
                            gl.glTranslatef(5.,0.,0.)
                            gl.glColor3f (1.0, 1.0, 0.0)
                            glu.gluSphere (quad, 1.0, 20, 20)

                            gl.glPopMatrix()

                            gl.glPopMatrix()
                            #gl.glFlush ()
                            gl.glFinish ()

                        @ctx.add
                        def resize(obj=obj):
                            (w,h) = obj.GetSize()
                            if not w or not h: 
                                return

                            #obj.SetCurrent()


                            gl.glViewport (0, 0, w, h)
                            gl.glMatrixMode (gl.GL_PROJECTION)
                            gl.glLoadIdentity ()
                            glu.gluPerspective(45, float(w)/h, 1, 100)
                            gl.glMatrixMode (gl.GL_MODELVIEW)
                            gl.glLoadIdentity ()
                            glu.gluLookAt (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0)
                            #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

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

                        CallAfter(obj.Refresh)
                    </opengl-canvas>
                    <layout layout-cfg='0, EXPAND'>
                        <button text='Enabled'><event>ctx.toggle()</event></button>
                    </layout>
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



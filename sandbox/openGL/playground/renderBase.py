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

import sys
import time

from TG.openGL.raw import gl
from TG.openGL.raw.gl import glFlush

from TG.skinning.toolkits.wx import wxSkinModel, XMLSkin

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

xmlSkin = XMLSkin("""<?xml version='1.0'?>
<skin xmlns='TG.skinning.toolkits.wx'>
    <style>
        frame {frame-main:1; locking:0; show: True}
        frame>layout {layout-cfg:1,EXPAND}
        frame>layout>panel {layout-cfg:1,EXPAND}

        opengl-canvas {
            layout-cfg:1,EXPAND; 
            gl-style:WX_GL_RGBA, WX_GL_DOUBLEBUFFER, WX_GL_DEPTH_SIZE, 8;
            }
    </style>

    <frame>
        ctx.model.setFrame(obj)
        <menubar>
            <menu text='View'>
                <item text='Full Screen\tCtrl-F' help='Shows My Frame on the entire screen'>
                    <event>
                        if ctx.frame.IsFullScreen():
                            ctx.frame.ShowFullScreen(False)
                        else:
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
                    <opengl-canvas ctxobj='canvas'>
                        ctx.model.canvas = obj
                        <timer ctxobj='model.repaintTimer' > 
                            refresh = ctx.model.refresh
                            canvas = ctx.canvas
                            <event>
                                refresh(canvas)
                            </event>
                        </timer>

                        initialize = ctx.model.initialize
                        refresh = ctx.model.refresh
                        resize = ctx.model.resize

                        initialize(obj)

                        <event>
                            resize(obj)
                        </event>
                        <event type="EVT_SIZE">
                            resize(obj)
                        </event>                        
                        <event type="EVT_ERASE_BACKGROUND"/>

                        <event type='EVT_MOUSE_EVENTS' run='ctx.model.onMouse(evt)' />
                        <!--<event type='EVT_KEY_UP' run='ctx.model.onKeyboard(evt)' />-->
                        <!--<event type='EVT_KEY_UP' run='ctx.model.onKeyboard(evt)' />-->
                        <event type='EVT_CHAR' run='ctx.model.onChar(evt)' />
                    </opengl-canvas>
                </layout>
            </panel>
        </layout>
        obj.SetClientSize(ctx.model.clientSize)
        <event>
            if ctx.model.onQuit():
                evt.Skip()
            else:
                evt.Veto()
        </event>
    </frame>
</skin>
""")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModelBase(wxSkinModel):
    xmlSkin = xmlSkin
    clientSize = (800, 800)
    frameTitle = 'GLCanvas'
    fpsFormat = '%.1f fps (refresh), %.0f fps (render)'
    fps = 60

    if sys.platform.startswith('win'):
        timestamp = staticmethod(time.clock)
    else: 
        timestamp = staticmethod(time.time)

    def setFrame(self, frame):
        self.frame = frame
        self.timingLog = []

    _lastUpdate = 0
    _historyKeep = 10
    _historyLen = 0
    def fpsUpdate(self, renderStart, renderEnd):
        self.timingLog.append(renderEnd - renderStart)

        timeDelta = renderEnd - self._lastUpdate
        if timeDelta >= 1 and len(self.timingLog):
            newEntries = (len(self.timingLog) - self._historyLen - 1)
            fpsRefresh = newEntries / timeDelta
            fpsRender = len(self.timingLog) / sum(self.timingLog, 0.0)

            self._updateFPSInfo(fpsRefresh, fpsRender)

            self.timingLog = self.timingLog[-self._historyKeep:]
            self._historyLen = len(self.timingLog)
            self._lastUpdate = renderEnd
            return True

    def _updateFPSInfo(self, fpsRefresh, fpsRender):
        fpsStr = self.fpsFormat % (fpsRefresh, fpsRender)
        self._printFPS(fpsStr)

    fpsStr = 'Waiting'
    def _printFPS(self, fpsStr):
        self.fpsStr = fpsStr
        print '\r', fpsStr.ljust(75),
        sys.stdout.flush()

    def onMouse(self, evt):
        pass

    def onChar(self, evt):
        pass

    def setFps(self, fps):
        self.fps = fps
        if self.fps > 0:
            self.repaintTimer.Start(1000./self.fps)
        else: 
            self.repaintTimer.Stop()
        del self.timingLog[:]

    def initialize(self, glCanvas):
        self.setFps(self.fps)

        self._lastUpdate = self.timestamp()
        glCanvas.SetCurrent()
        self.renderInit(glCanvas, self._lastUpdate)
        self.resize(glCanvas)

    def renderInit(self, glCanvas, renderStart):
        pass

    def resize(self, glCanvas):
        glCanvas.SetCurrent()
        self.renderResize(glCanvas)
        self.refresh(glCanvas)

    def renderResize(self, glCanvas):
        pass

    def refresh(self, glCanvas):
        self.renderSwap(glCanvas)
        t0 = self.timestamp()
        self.renderContent(glCanvas, t0)
        t1 = self.timestamp()
        self.renderFinish(glCanvas, t0, t1)

    def renderSwap(self, glCanvas):
        glCanvas.SetCurrent()
        glCanvas.SwapBuffers()

    def renderContent(self, glCanvas, timestamp):
        pass

    def renderFinish(self, glCanvas, timestamp, timestampEnd):
        ## Note: Don't use unless absolutely needed:
        ## glFinish () # VERY expensive

        self.fpsUpdate(timestamp, timestampEnd)
        glFlush()
    
    def onQuit(self):
        self.repaintTimer.Stop()
        return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



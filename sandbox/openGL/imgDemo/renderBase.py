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

from TG.openGL.raw.gl import glGetError
from TG.openGL.raw.glu import gluErrorString

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
    fpsFormat = '%.1f fps (true), %.0f fps (%.5f:%.5f render:swap)'
    fps = 60

    if sys.platform.startswith('win'):
        timestamp = staticmethod(time.clock)
    else: 
        timestamp = staticmethod(time.time)

    def setFrame(self, frame):
        self.frame = frame
        self.timingLogRender = []
        self.timingLogSwap = []
        self.timingLogEnds = []

    _lastUpdate = 0
    _historyKeep = 15
    _historyLen = 0
    def fpsUpdate(self, renderStart, renderEnd, renderFinish):
        self.timingLogRender.append(renderEnd - renderStart)
        self.timingLogSwap.append(renderFinish - renderEnd)

        timeDelta = renderFinish - self._lastUpdate
        totalEntries = len(self.timingLogRender)
        if timeDelta >= 1 and totalEntries:
            newEntries = (totalEntries - self._historyLen)
            fpsTrue = newEntries / timeDelta
            timeRender = sum(self.timingLogRender, 0.0)
            #fpsRender = totalEntries / timeRender
            timeSwap = sum(self.timingLogSwap, 0.0)
            #fpsSwap = totalEntries / timeSwap
            fpsEffective = totalEntries / (timeRender + timeSwap)

            self._updateFPSInfo(fpsTrue, fpsEffective, timeRender, timeSwap, totalEntries)

            self.timingLogRender[:] = self.timingLogRender[-self._historyKeep:]
            self.timingLogSwap[:] = self.timingLogSwap[-self._historyKeep:]
            self._historyLen = len(self.timingLogRender)
            self._lastUpdate = renderFinish

    def _updateFPSInfo(self, fpsTrue, fpsEffective, timeRender, timeSwap, totalEntries):
        fpsStr = self.fpsFormat % (fpsTrue, fpsEffective, timeRender/totalEntries, timeSwap/totalEntries)
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

    def initialize(self, glCanvas):
        if self.fps:
            self.repaintTimer.Start(1000./self.fps)
        else: self.repaintTimer.Stop()
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
        renderStart = self.timestamp()
        glCanvas.SetCurrent()
        self.renderContent(glCanvas, renderStart)
        renderEnd = self.timestamp()
        self.renderFinish(glCanvas, renderStart, renderEnd)
        renderFinish = self.timestamp()
        self.fpsUpdate(renderStart, renderEnd, renderFinish)

    def renderContent(self, glCanvas, renderStart):
        pass

    def renderFinish(self, glCanvas, renderStart, renderEnd):
        ## Note: Don't use unless absolutely needed:
        ## glFinish () # VERY expensive

        ##glFlush () # implicit in SwapBuffers
        glCanvas.SwapBuffers()
    
    def onQuit(self):
        self.repaintTimer.Stop()
        return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



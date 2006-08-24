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

import array
from string import printable

from ctypes import byref, c_void_p

from renderBase import *

from TG.freetype2.face import FreetypeFace

from TG.openGL.raw import gl, glu
from TG.openGL.raw.gl import *
from TG.openGL.raw.glu import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Constants / Variables / Etc. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class RenderSkinModel(RenderSkinModelBase):
    fonts = {
            'Arial':'/Library/Fonts/Arial',
            'Zapfino':'/Library/Fonts/Zapfino.dfont',
            'Monaco':'/System/Library/Fonts/Monaco.dfont',
            'AppleGothic':'/System/Library/Fonts/AppleGothic.dfont',
            'LucidaGrande':'/System/Library/Fonts/LucidaGrande.dfont',
            }

    def glCheck(self):
        glErr = glGetError()
        if glErr:
            raise Exception("GL Error: 0x%x" % glErr)
        return True

    def renderInit(self, glCanvas, renderStart):
        glClearColor(0.0, 0.0, 0.0, 0.0)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)

        glShadeModel(GL_SMOOTH)
        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)

        #glPolygonMode(GL_FRONT, GL_FILL)
        #glPolygonMode(GL_BACK, GL_LINE)
        #glCullFace(GL_BACK)
        #glEnable(GL_CULL_FACE)

        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if 0:
            self.quad = gluNewQuadric()
            if 1: gluQuadricDrawStyle(self.quad, GLU_FILL)
            elif 1: gluQuadricDrawStyle(self.quad, GLU_LINE)
            elif 1: gluQuadricDrawStyle(self.quad, GLU_POINT)
            gluQuadricNormals(self.quad, GLU_SMOOTH)
            gluQuadricTexture(self.quad, True)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        self.loadCheckerBoard()
        self.loadFontTexture()

        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    def loadCheckerBoard(self):
        self.checkerBoardTexName = GLenum(0)
        return False

        textureFormat = GL_INTENSITY
        dataFormat = GL_LUMINANCE
        itemSize = GL_UNSIGNED_BYTE
        iSize = 1 << 8; jSize = iSize
        checkerBoard = array.array('B', [e for i in xrange(iSize) for j in xrange(jSize) 
                                            for c in (0xff & (((i>>3) + (j>>3)) & 1) * 255,)
                                                for e in (c,)])
        texptr = checkerBoard.buffer_info()[0]

        glGenTextures(1, byref(self.checkerBoardTexName))
        glBindTexture(GL_TEXTURE_2D, self.checkerBoardTexName)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        if iSize <= 32:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            glTexImage2D(GL_TEXTURE_2D, 0, textureFormat, iSize, jSize, 0, dataFormat, itemSize, texptr)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
            gluBuild2DMipmaps(GL_TEXTURE_2D, textureFormat, iSize, jSize, dataFormat, itemSize, texptr)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def loadFontTexture(self):
        fontSize = 64
        width, height = 1024, fontSize*16

        self.fontTexture = GLenum(0)
        glGenTextures(1, byref(self.fontTexture))
        glBindTexture(GL_TEXTURE_2D, self.fontTexture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        checkerBoard = array.array('B', [0]*width*height)
        texptr = checkerBoard.buffer_info()[0]

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_INTENSITY, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, texptr)
        self.glCheck()

        fontFilename = self.fonts['LucidaGrande']
        fontFilename = self.fonts['Zapfino']
        print fontFilename
        self.ftFace = FreetypeFace(fontFilename)
        self.ftFace.setPixelSize(fontSize)
        metrics = self.ftFace.size[0].metrics
        maxFontHeight = (metrics.ascender - metrics.descender) >> 6
        maxFontWidth = (metrics.max_advance)>>6
        print maxFontWidth, maxFontHeight
        #import sys
        #sys.exit(0)

        advances = []
        widths = []
        heights = []

        x = 1
        y = 1
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0)
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        maxRowHeight = 0
        for char, glyph in self.ftFace.iterChars(printable):
            #print (char, glyph)
            advances.append(glyph.advance.x >> 6)
            widths.append(glyph.bitmap.width)
            heights.append(glyph.bitmap.rows)

            bitmap = glyph.bitmap
            assert bitmap.num_grays == 256, bitmap.num_grays
            w = bitmap.width
            h = bitmap.rows
            maxRowHeight = max(h, maxRowHeight)
            #dy = glyph.bitmapTop - h

            if x+w > width:
                x = 1
                y += maxRowHeight + 1
                maxRowHeight = h

            glPixelStorei(GL_UNPACK_SKIP_PIXELS, bitmap.pitch-w)
            glTexSubImage2D(GL_TEXTURE_2D, 0, x, y, w, h, GL_LUMINANCE, GL_UNSIGNED_BYTE, bitmap.buffer) 
            x += w + 1

        if 1:
            from collections import defaultdict

            advancesFreq = defaultdict(lambda: 0)
            for e in advances: advancesFreq[e]+=1
            advancesFreq = ', '.join('%s:%s' % e for e in advancesFreq.iteritems())

            widthsFreq = defaultdict(lambda: 0)
            for e in widths: widthsFreq[e]+=1
            widthsFreq = ', '.join('%s:%s' % e for e in widthsFreq.iteritems())

            heightsFreq = defaultdict(lambda: 0)
            for e in heights: heightsFreq[e]+=1
            heightsFreq = ', '.join('%s:%s' % e for e in heightsFreq.iteritems())

            print sum(advances), (min(advances), max(advances)), advancesFreq
            print sum(widths), (min(widths), max(widths)), widthsFreq
            print sum(heights), (min(heights), max(heights)), heightsFreq

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sized = False
    def renderResize(self, glCanvas):
        (w,h) = glCanvas.GetSize()
        if not w or not h: return
        #print 'resize:', (w, h)

        glViewport (0, 0, w, h)
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()

        ho = 20./h
        wo = 20./w
        #gluOrtho2D(-1-wo, 1+wo, -1-ho, 1+ho)
        gluOrtho2D(0, w, 0, h)
        self.viewLR = 0, w
        self.viewBT = 0, h

        #gluPerspective(60, float(w)/h, 1, 100)
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()
        #gluLookAt (0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 0.0)
        #glTranslatef(0.,0.,10.)
        self.sized = True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def renderContent(self, glCanvas, renderStart):
        glClear (GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if not self.sized:
            return

        glEnable(GL_TEXTURE_2D)
        glPushMatrix()

        c = renderStart*5
        #glRotatef((c*4) % 360.0, 0, 0, 1)
        #glRotatef((c*3) % 360.0, 0, 1, 0)
        #glRotatef((c*5) % 360.0, 0, 1, 0)

        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
        if self.checkerBoardTexName:
            glBindTexture(GL_TEXTURE_2D, self.checkerBoardTexName)
        else:
            glBindTexture(GL_TEXTURE_2D, self.fontTexture)

        vl, vr = self.viewLR
        vb, vt = self.viewBT
        glBegin(GL_QUADS)

        glVertex3s(vl, vt, 0)
        glTexCoord2s(0, 1)
        glColor3f(0.5, 1, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vl, vb, 0)
        glTexCoord2s(1, 1)
        glColor3f(0.5, 0.5, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vr, vb, 0)
        glTexCoord2s(1, 0)
        glColor3f(1, 0.5, 1)
        glNormal3s(0, 0, 1)

        glVertex3s(vr, vt, 0)
        glTexCoord2s(0, 0)
        #glColor3f(1, 1, 1)
        glNormal3s(0, 0, 1)

        glEnd()

        glPopMatrix()

        glDisable(GL_TEXTURE_2D)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    RenderSkinModel().skinModel()


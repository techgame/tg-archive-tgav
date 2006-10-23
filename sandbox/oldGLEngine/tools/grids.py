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

import Image
from OpenGL import GL, GLU

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _DecodeImageData(strdata, size):
    rawdata = strdata.decode('base64').decode('zlib') 
    return rawdata, size
def _EncodeImageData(filename):
    image = Image.open(filename)
    rawdata = image.tostring('raw', 'RGBX', 0, -1)
    strdata = rawdata.encode('zlib').encode('base64') 
    return strdata, image.size

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rawdata = _DecodeImageData("""
eJzt3c9qE1EYxuFb76KLUqpBoa0RaRQk1sXERdxIsvEi+tfWaimiV3HMK8zYrW4C8z0HzgX8zsOZ
SWbztWZVX+tfH9vyx6J1D+/b6f28zb+/HeVOWxrTmuah//NDW366b93yWzv9cNfm3ddR7rSlMa1p
7lfOI2czu3vdprev2vHNy1HutKUxrWke+jfnkbOZvbtt0zdf2vFsnDttaUxrmvuVO5FzObqZtsn1
YTu4ej7KnbY0pjXNQ//mTuRcjk6u2+TFVTs4vBzlTlsa05rmfuU+5G7kfPYuJ233/Mkod9rSmNY0
D/2b+5C7kfPZe3bRdp+ej3KnLY1pTXO/8m7M8zF3JOe0c7Y/yp22NKY1zUP/5t2Y52PuSM5pZ/9s
lDttaUxrmvnz58+fP3/+/Pnz589/2078+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/
f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/7bduLPnz9//vz58+fPnz9//vz5
8+fPnz9//vz58+fPnz////U3/632/DfzH2vPf6w5/7X72198/mu9+c9dW/1cDv2r4vOfLcuqu9YF
n3+rR8+/vAvyTqjz/lv8eef3q/rvn9jX+/27GPqr//+p/v+3+veP6t+/qn//9P2bP3/+/Pnz579t
J/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78+fPnz58/f/78
+fPnz58/f/78+fPnz58//2078efPnz9//vz58+fPnz9//vz58+fPnz9//vz58+fPnz9//vz58/83
f/Pfas9/M/+x9vxH818Xpee/Vpz/vH40/zmzkHMeVec/W3XXb80sVlc=

""", (128,128))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextureGrid(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Constants / Variables / Etc. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    GridExtent = 1
    TextureExtent = 10
    TextureID = None
    ReloadTexture = 0
    imagedata, imagesize = Rawdata

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Public Methods 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, **kw):
        for name, value in kw.iteritems():
            setattr(self, name, value)

    def LoadGridTexture(self, filename):
        image = Image.open(filename)
        self.imagedata = image.tostring('raw', 'RGBX', 0, -1)
        self.imagesize = image.size
        self.ReloadTexture = 1

    def _LoadTexture(self):
        w, h = self.imagesize

        if self.TextureID is None:
            self.TextureID = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureID)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, 3, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, self.imagedata)
        #GLU.gluBuild2DMipmaps(GL.GL_TEXTURE_2D, 3, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, self.imagedata)

        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        #GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_NEAREST)

    def GLExecute(self, context):
        if self.ReloadTexture or self.TextureID is None:
            self._LoadTexture()
            self.ReloadTexture = 0

        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.TextureID)
        GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_REPLACE)

        GridExtent = self.GridExtent
        TextureExtent = self.TextureExtent

        GL.glColor3f(1, 1., 1.)
        GL.glNormal3f(0, 1., 0.)

        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0)
        GL.glVertex3f(-GridExtent, 0., -GridExtent)
        GL.glTexCoord2f(0, TextureExtent)
        GL.glVertex3f(-GridExtent, 0., GridExtent)
        GL.glTexCoord2f(TextureExtent, TextureExtent)
        GL.glVertex3f(GridExtent, 0., GridExtent)
        GL.glTexCoord2f(TextureExtent, 0)
        GL.glVertex3f(GridExtent, 0., -GridExtent)
        GL.glEnd()

        GL.glDisable(GL.GL_TEXTURE_2D)

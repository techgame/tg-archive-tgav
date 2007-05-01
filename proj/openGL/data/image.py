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

from numpy import array

from TG.geomath.data.box import Box

from ..raw import gl, glext
from .texture import Texture

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def imagePremultiply(image, raiseOnInvalid=True):
    bands = image.getbands()
    if bands[-1] != 'A':
        if raiseOnInvalid:
            raise TypeError("Image does not have an alpha channel as the last band")
        else: 
            return image

    imageData = image.getdata()

    a = imageData.getband(len(bands)-1)
    
    for idx in xrange(len(bands)-1):
        ba = a.chop_multiply(imageData.getband(idx))
        imageData.putband(ba, idx)

    return image

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ImageTexture(Texture):
    texParams = Texture.texParams + [
            ('target', ('rect', '2d')),
            ('wrap', gl.GL_CLAMP),
            ('genMipmaps', True),
            ('magFilter', gl.GL_LINEAR),
            ('minFilter', gl.GL_LINEAR),
            ('minFilter', gl.GL_LINEAR_MIPMAP_LINEAR),
            ]

    modeFormatMap = {
        'RGBA': (gl.GL_RGBA, gl.GL_UNSIGNED_BYTE),
        'RGB': (gl.GL_RGB, gl.GL_UNSIGNED_BYTE),
        'LA': (gl.GL_LUMINANCE_ALPHA, gl.GL_UNSIGNED_BYTE),
        'L': (gl.GL_LUMINANCE, gl.GL_UNSIGNED_BYTE),
        }

    box = Box.property(publish='box')

    def __init__(self, image=None):
        self.image = image
        self.send()

    _image = None
    def getImage(self):
        return self._image
    def setImage(self, image, format=True):
        if format is True:
            self.format = self.modeFormatMap[image.mode][0]
        elif format is not None:
            self.format = format

        self._image = image
        self.box.size = image.size
    image = property(getImage, setImage)

    def premultiply(self, check=True):
        imagePremultiply(self.image, check)

    def send(self, sgo=None):
        image = self.image
        self.select()
        dataFormat, dataType = self.modeFormatMap[image.mode]
        size = self.asValidSize(image.size)
        data = self.data2d(size=size, format=dataFormat, dataType=dataType)
        data.setImageOn(self)

        data.texString(image.tostring(), dict(alignment=1,))
        data.setSubImageOn(self, size=image.size)


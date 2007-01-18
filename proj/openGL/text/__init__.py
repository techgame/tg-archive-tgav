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

from font import Font, Font2d
import freetypeFontLoader

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loaderFromFaceFilename(klass, face, size=None, **kw):
    loader = freetypeFontLoader.FreetypeFontLoader(face, size, **kw)
    loader.FontFactory = klass
    return loader

def fromFaceFilename(klass, face, size=None, **kw):
    loader = loaderFromFaceFilename(klass, face, size, **kw)
    return loader.compile()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Font.fromFace = classmethod(fromFaceFilename)
Font.fromFilename = classmethod(fromFaceFilename)
Font.loaderFromFace = classmethod(loaderFromFaceFilename)
Font.loaderFromFilename = classmethod(loaderFromFaceFilename)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Font2d.fromFace = classmethod(fromFaceFilename)
Font2d.fromFilename = classmethod(fromFaceFilename)
Font2d.loaderFromFace = classmethod(loaderFromFaceFilename)
Font2d.loaderFromFilename = classmethod(loaderFromFaceFilename)

fromFace = Font2d.fromFace
fromFilename = Font2d.fromFilename


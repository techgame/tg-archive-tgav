#!/usr/bin/env python
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

import sys
from array import array
import ctypes
from ctypes import byref

from TG.freetype2.face import FreetypeFace

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check(err):
    if err: raise RuntimeError("Error: %s" % (err,))

class Bitmap(object):
    def __init__(self, char, x, y, ax, bitmap):
        self.char = char
        self.bitmap = bitmap
        self.h = len(bitmap)
        self.w = self.h and len(bitmap[0])
        self.x = x
        self.y = y
        self.ax = ax
        self.dy = y - self.h

        if char.isspace():
            asciimap = " "
        else:
            asciimap = " :|" + char
        self.ascii = getBitmapASCII(bitmap, asciimap)
        self.image = [array('B', line).tostring() for line in bitmap]
        assert len(self.image) == len(bitmap)
        if bitmap:
            assert len(self.image[0]) == len(bitmap[0])

    def __str__(self):
        return '\n'.join(self.ascii)

def getBitmapFromGlyph(char, glyph):
    ax = glyph.advance.x >> 6
    return Bitmap(char, glyph.bitmapLeft, glyph.bitmapTop, ax, getBitmap(glyph.bitmap))

def getBitmap(bm):
    assert bm.num_grays == 256, bm.num_grays
    pitch = bm.pitch; buf = bm.buffer
    data = [buf[r*bm.pitch:(r+1)*pitch] for r in xrange(bm.rows)]
    return data

asciimap = ' 123456789abcdef'
def getBitmapASCII(bitmap, asciimap=asciimap, inner=''.join):
    d = 256 / len(asciimap)
    return [inner(asciimap[i/d] for i in l) for l in bitmap]

def renderASCII(str, fontMap):
    bmaps = [(bm, bm.ascii) for bm in (fontMap[c] for c in str)]
    return _render(bmaps, None, ' ')

def renderImage(str, fontMap):
    bmaps = [(bm, bm.image) for bm in (fontMap[c] for c in str)]
    return _render(bmaps, None, '\x00')

def _render(bmaps, sep='\n', pad=' '):
    baseline = max(bm.y for bm,image in bmaps)
    desc = min(bm.dy for bm,image in bmaps)
    height = baseline - desc

    result = [''] * height
    for bm, image in bmaps:
        prepad = bm.x * pad
        blank = (bm.x + bm.w) * pad
        postpad = (bm.ax - (bm.x + bm.w)) * pad
        blank+=postpad

        py = -1
        for py in xrange(baseline-bm.y):
            result[py] += blank

        for line in image:
            py += 1
            line = prepad + line + postpad

            result[py] += line

        for py in xrange(py+1, height):
            result[py] += blank

    if sep is not None:
        result = sep.join(result)
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    if 1:
        faceFilename = '/Library/Fonts/Arial'
        faceFilename = '/Library/Fonts/Zapfino.dfont'
        faceFilename = '/System/Library/Fonts/Monaco.dfont'
        faceFilename = '/System/Library/Fonts/AppleGothic.dfont'
        faceFilename = '/System/Library/Fonts/LucidaGrande.dfont'
    print faceFilename
    face = FreetypeFace(faceFilename)

    try: 
        size = int(sys.argv[1])
    except LookupError:
        size = int(raw_input('Size [20]> ') or 20)

    imageDPI = int(raw_input("Image DPI [0 for Text]> ") or 0)
    if imageDPI:
        face.setCharSize(size*64, imageDPI)
    else:
        face.setPixelSize(size)

    fontBaseline = int(face.ascender)
    fontHeight = fontBaseline + (-face.descender)
    fontBaseline >>= 6
    fontHeight >>= 6

    fontMap = {}

    def loadChar(char):
        if char not in fontMap:
            fontMap[char] = getBitmapFromGlyph(char, face.loadChar(char))
        return fontMap[char]

    import string
    #loadChar(' ')
    map(loadChar, string.printable)

    while 1:
        bFlip = int(raw_input("Flip [0]> ") or 0)
        try:
            text = raw_input(u'Text> ').decode('utf-8')
        except EOFError:
            break

        if not text.strip():
            break

        map(loadChar, text)

        if imageDPI:
            img = renderImage(text, fontMap)
        else:
            img = renderASCII(text, fontMap)

        if bFlip:
            img = [''.join(l[::-1]) for l in zip(*img)]

        if imageDPI:
            print len(img[0]), len(img)
            imgFileName = '%dx%d-%s.raw' % (len(img[0]), len(img), text)
            imgFile = file(imgFileName, 'wb')
            for l in img:
                imgFile.write(''.join(l))
            imgFile.close()

        else:
            for l in img:
                print l.encode('utf-8')


#!/usr/bin/env python
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

import os
import sys
import time

from struct import unpack

from TG.common.utilities import textWiggler
from TG.audioFormats.waveFile import WaveFormat
from TG import openAL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

wig = textWiggler()

context = openAL.newContext()
src = openAL.Source()
buf = openAL.Buffer()

def play(buf):
    src.play(buf)
    while src.isPlaying():
        print wig.next(),
        time.sleep(0.1)

def main(fn, bWrite='0'):
    wr = WaveFormat(fn)

    if not int(bWrite):
        print 'playing:', wr
        data = wr.readFrames()
        print 'raw:', data[:40].encode('hex')
        wr.rewind()
        buf.setDataFromWave(wr)
        play(buf)

    if int(bWrite):
        wr.rewind()
        ww = WaveFormat('mulaw-'+fn, 'wb')
        ww.copyFormat(wr, waveFormat=ww.formatEnum.WAVE_FORMAT_MULAW)
        ww.writeFrames(wr.readFrames())
        ww.close()

    if 0 and int(bWrite):
        wr.rewind()
        ww = WaveFormat('adpcm-'+fn, 'wb')
        #ww.copyFormat(wr, waveFormat=ww.formatEnum.WAVE_FORMAT_DVI_ADPCM)
        ww.copyFormat(wr, waveFormat=ww.formatEnum.WAVE_FORMAT_ADPCM)
        ww.writeFrames(wr.readFrames())
        ww.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    main(*sys.argv[1:])



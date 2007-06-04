#!/usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import sys
import time

from itertools import takewhile, count
from TG.common.utilities import textWiggler

from TG.audioFormats.waveFile import WaveFormat

from TG import openAL

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loadWave(name, src):
    buf = openAL.Buffer.fromFilename(name)
    src.queue(buf)
    return buf

def main():
    wig = textWiggler()

    device = openAL.Device()
    context = openAL.Context(device, None)

    if 1:
        print "OpenAL Library:"
        print '  version:', openAL.library.version
        print '  vendor:', openAL.library.vendor
        print '  renderer:', openAL.library.renderer
        print '  extensions:', openAL.library.extensions

    if 1 and openAL.Capture is not None:
        print 'Capture:'
        print '  default:', openAL.Capture.defaultDeviceName()
        for each in openAL.Capture.allDeviceNames():
            print '    available:', each
        print
        capture = openAL.Capture(count=1<<20)
        print
        print capture.name
        print

    buf = openAL.Buffer()
    while 1:

        if raw_input('Ready? > ').lower() != 'y':
            break

        print capture
        print capture.frequency, capture.channels, capture.width

        wave = WaveFormat('capture.wav', 'wb')
        wave.setFormat(wave.formatEnum.WAVE_FORMAT_MULAW, capture.frequency, capture.channels, capture.width)
        #wave.setFormat(wave.formatEnum.WAVE_FORMAT_PCM, capture.frequency, capture.channels, capture.width)
        wave.writeWaveHeader()

        sampleList = []

        @capture.kvo('sampleCount')
        def onCountChange(capture, sampleCount):
            if sampleCount <= 0:
                print '.',
                return

            print 'Capture samples available:', sampleCount
            data = capture.samples()
            assert len(data) > 0
            wave.writeFrames(data)
            sampleList.append(data)

        capture.start()
        for x in xrange(60):
            time.sleep(0.1)
            context.process()

        capture.stop()
        wave.close()

        print
        data = ''.join(sampleList)
        print 'all data:', len(data), len(data)/float(capture.frequency*capture.channels*capture.width)
        print 'raw:', data[:40].encode('hex')
        buf.setDataFromCapture(data, capture)

        print 'Queing audio:'
        src = openAL.Source(buf)

        @src.kvo('sec_offset')
        def onCountChange(src, s):
            print "Second Offset:", s

        while src.isPlaying():
            context.process()
            print wig.next(),
            sys.stdout.flush()
            time.sleep(0.1)

        break

if __name__=='__main__':
    main()


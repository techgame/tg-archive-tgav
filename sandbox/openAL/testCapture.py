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
        capture = openAL.Capture()
        print
        print capture.name
        print

    buf = openAL.Buffer()
    while 1:

        if raw_input('Ready? [y]> ').lower() not in ['y', 'yes', '']:
            break

        print capture
        print capture.frequency, capture.channels, capture.width

        capture.wave = WaveFormat('capture.wav', 'wb')
        capture.wave.setFormatFrom('PCM', capture)
        capture.wave.writeWaveHeader()
        capture.lemming = 0

        counts = []
        sampleList = []

        @capture.kvo('sampleCount')
        def onCountChange(capture, sampleCount):
            if sampleCount <= 0:
                print '.',
                return

            data = capture.samples()
            capture.wave.writeFrames(data)
            sampleList.append(data)
            print 'Capture samples available:', sampleCount, 'bytes:', len(data)

        capture.start()
        for x in xrange(60):
            time.sleep(0.1)
            context.process()

        capture.stop()
        capture.wave.close()

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

        @src.kvo('state')
        def onStateChange(src, s):
            print "State:", s


        while src.isPlaying():
            context.process()
            print wig.next(),
            sys.stdout.flush()
            time.sleep(0.1)

        for x in xrange(10):
            context.process()
            time.sleep(0.1)

        break

if __name__=='__main__':
    main()


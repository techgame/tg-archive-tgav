#!/usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os
import sys
import time

from itertools import takewhile
from TG.common.utilities import textWiggler

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

    print 'Playback:'
    print '  default:', openAL.Device.defaultDeviceName()
    for each in openAL.Device.allDeviceNames():
        print '    available:', each
    print

    device = openAL.Device()
    context = openAL.Context(device)
    context.makeCurrent()

    print 'device:', device
    print '  name:', device.name
    print '  version:', device.version

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
        captureDevice = openAL.Capture()

    srcs = openAL.SourceCollection(2)
    loadWave('Footsteps.wav', srcs[0])
    loadWave('sound_monkey.wav', srcs[1])

    srcs.play()
    n = 0

    if 1:
        srcs[0].velocity = (-0.1,0,0)
        srcs[1].velocity = (0,0.3,0)

    while srcs.isPlaying():
        print wig.next(),
        time.sleep(0.1)
        if 1:
            srcs[0].position = (0., 0.5-0.01*n, -1.)
            print srcs[0].position
            print '(%1.1f, %1.1f, %1.1f)' % tuple(srcs[0].position),

            srcs[1].position = (-1 + 0.1*n, 0., 1.)
            print '(%1.1f, %1.1f, %1.1f)' % tuple(srcs[1].position),

            print
            sys.stdout.flush()
            n+=1

if __name__=='__main__':
    main()


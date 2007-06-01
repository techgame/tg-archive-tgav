#!/usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import time
from TG.cepstral.engine import CepstralEngine

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(argv):
    engine = CepstralEngine()
    port = engine.newPort()
    port.setVoiceName('Diane')

    port.speak(' '.join(argv))

    @port.events.on('word')
    def onTTSWord(evt):
        print 'tts word:', evt.text

    @port.events.on(None)
    def onTTS(evt):
        if evt.name == 'phoneme':
            return
        print evt, evt.textPos

    while port.status() != 'done':
        print port.status()
        time.sleep(1)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    main(sys.argv[1:])


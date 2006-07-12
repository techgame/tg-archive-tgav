#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *
from altypes import *
from alctypes import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alut.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALUTAPI = None # empty
ALUTAPIENTRY = None # empty

#~ line: 39, skipped: 14 ~~~~~~

@bind(ALvoid, [POINTER(ALint), POINTER(POINTER(ALbyte))])
def alutInit(argc, argv): pass

@bind(ALvoid, [])
def alutExit(): pass

@bind(ALvoid, [POINTER(ALbyte), POINTER(ALenum), POINTER(POINTER(ALvoid)), POINTER(ALsizei), POINTER(ALsizei)])
def alutLoadWAVFile(file, format, data, size, freq): pass

@bind(ALvoid, [POINTER(ALbyte), POINTER(ALenum), POINTER(POINTER(ALvoid)), POINTER(ALsizei), POINTER(ALsizei)])
def alutLoadWAVMemory(memory, format, data, size, freq): pass

@bind(ALvoid, [ALenum, POINTER(ALvoid), ALsizei, ALsizei])
def alutUnloadWAV(format, data, size, freq): pass



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alut.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


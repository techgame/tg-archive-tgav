#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *
from altypes import *
from alctypes import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alc.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALCAPI = None # empty
ALCAPIENTRY = None # empty
ALCdevice = ALCvoid # typedef ALCdevice
ALCcontext = ALCvoid # typedef ALCcontext

#~ line: 34, skipped: 5 ~~~~~~

@bind(POINTER(ALCubyte), [POINTER(ALCdevice), ALCenum])
def alcGetString(device, param): pass

@bind(ALCvoid, [POINTER(ALCdevice), ALCenum, ALCsizei, POINTER(ALCint)])
def alcGetIntegerv(device, param, size, data): pass


@bind(POINTER(ALCdevice), [POINTER(ALCubyte)])
def alcOpenDevice(deviceName): pass

@bind(ALCvoid, [POINTER(ALCdevice)])
def alcCloseDevice(device): pass


@bind(POINTER(ALCcontext), [POINTER(ALCdevice), POINTER(ALCint)])
def alcCreateContext(device, attrList): pass

@bind(ALCboolean, [POINTER(ALCcontext)])
def alcMakeContextCurrent(context): pass

@bind(ALCvoid, [POINTER(ALCcontext)])
def alcProcessContext(context): pass

@bind(POINTER(ALCcontext), [])
def alcGetCurrentContext(): pass

@bind(POINTER(ALCdevice), [POINTER(ALCcontext)])
def alcGetContextsDevice(context): pass

@bind(ALCvoid, [POINTER(ALCcontext)])
def alcSuspendContext(context): pass

@bind(ALCvoid, [POINTER(ALCcontext)])
def alcDestroyContext(context): pass


@bind(ALCenum, [POINTER(ALCdevice)])
def alcGetError(device): pass


@bind(ALCboolean, [POINTER(ALCdevice), POINTER(ALCubyte)])
def alcIsExtensionPresent(device, extName): pass

@bind(POINTER(ALCvoid), [POINTER(ALCdevice), POINTER(ALCubyte)])
def alcGetProcAddress(device, funcName): pass

@bind(ALCenum, [POINTER(ALCdevice), POINTER(ALCubyte)])
def alcGetEnumValue(device, enumName): pass



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alc.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


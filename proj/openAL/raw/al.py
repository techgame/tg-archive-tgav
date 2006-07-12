#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *
from altypes import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/al.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALAPI = None # empty
ALAPIENTRY = None # empty
AL_CALLBACK = None # empty

#~ line: 62, skipped: 16 ~~~~~~

@bind(ALvoid, [ALenum])
def alEnable(capability): pass

@bind(ALvoid, [ALenum])
def alDisable(capability): pass

@bind(ALboolean, [ALenum])
def alIsEnabled(capability): pass


@bind(ALvoid, [ALenum, ALenum])
def alHint(target, mode): pass


@bind(ALboolean, [ALenum])
def alGetBoolean(param): pass

@bind(ALint, [ALenum])
def alGetInteger(param): pass

@bind(ALfloat, [ALenum])
def alGetFloat(param): pass

@bind(ALdouble, [ALenum])
def alGetDouble(param): pass

@bind(ALvoid, [ALenum, POINTER(ALboolean)])
def alGetBooleanv(param, data): pass

@bind(ALvoid, [ALenum, POINTER(ALint)])
def alGetIntegerv(param, data): pass

@bind(ALvoid, [ALenum, POINTER(ALfloat)])
def alGetFloatv(param, data): pass

@bind(ALvoid, [ALenum, POINTER(ALdouble)])
def alGetDoublev(param, data): pass

@bind(POINTER(ALubyte), [ALenum])
def alGetString(param): pass


@bind(ALvoid, [ALenum, ALint])
def alSetInteger(pname, value): pass

@bind(ALvoid, [ALenum, ALdouble])
def alSetDouble(pname, value): pass


#~ line: 87, skipped: 6 ~~~~~~

@bind(ALenum, [])
def alGetError(): pass


#~ line: 95, skipped: 8 ~~~~~~

@bind(ALboolean, [POINTER(ALubyte)])
def alIsExtensionPresent(fname): pass


#~ line: 103, skipped: 8 ~~~~~~

@bind(POINTER(ALvoid), [POINTER(ALubyte)])
def alGetProcAddress(fname): pass


#~ line: 110, skipped: 7 ~~~~~~

@bind(ALenum, [POINTER(ALubyte)])
def alGetEnumValue(ename): pass


#~ line: 130, skipped: 20 ~~~~~~

@bind(ALvoid, [ALenum, ALint])
def alListeneri(param, value): pass


#~ line: 137, skipped: 7 ~~~~~~

@bind(ALvoid, [ALenum, ALfloat])
def alListenerf(param, value): pass


#~ line: 145, skipped: 8 ~~~~~~

@bind(ALvoid, [ALenum, ALfloat, ALfloat, ALfloat])
def alListener3f(param, v1, v2, v3): pass


#~ line: 154, skipped: 9 ~~~~~~

@bind(ALvoid, [ALenum, POINTER(ALfloat)])
def alListenerfv(param, values): pass


@bind(ALvoid, [ALenum, POINTER(ALint)])
def alGetListeneri(param, value): pass

@bind(ALvoid, [ALenum, POINTER(ALfloat)])
def alGetListenerf(param, value): pass

@bind(ALvoid, [ALenum, POINTER(ALfloat), POINTER(ALfloat), POINTER(ALfloat)])
def alGetListener3f(param, v1, v2, v3): pass

@bind(ALvoid, [ALenum, POINTER(ALfloat)])
def alGetListenerfv(param, values): pass


#~ line: 174, skipped: 15 ~~~~~~

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alGenSources(n, sources): pass


@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alDeleteSources(n, sources): pass


@bind(ALboolean, [ALuint])
def alIsSource(id): pass


@bind(ALvoid, [ALuint, ALenum, ALint])
def alSourcei(source, param, value): pass

@bind(ALvoid, [ALuint, ALenum, ALfloat])
def alSourcef(source, param, value): pass

@bind(ALvoid, [ALuint, ALenum, ALfloat, ALfloat, ALfloat])
def alSource3f(source, param, v1, v2, v3): pass

@bind(ALvoid, [ALuint, ALenum, POINTER(ALfloat)])
def alSourcefv(source, param, values): pass


@bind(ALvoid, [ALuint, ALenum, POINTER(ALint)])
def alGetSourcei(source, param, value): pass

@bind(ALvoid, [ALuint, ALenum, POINTER(ALfloat)])
def alGetSourcef(source, param, value): pass

@bind(ALvoid, [ALuint, ALenum, POINTER(ALfloat), POINTER(ALfloat), POINTER(ALfloat)])
def alGetSource3f(source, param, v1, v2, v3): pass

@bind(ALvoid, [ALuint, ALenum, POINTER(ALfloat)])
def alGetSourcefv(source, param, values): pass


@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alSourcePlayv(n, sources): pass

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alSourcePausev(n, sources): pass

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alSourceStopv(n, sources): pass

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alSourceRewindv(n, sources): pass


@bind(ALvoid, [ALuint])
def alSourcePlay(source): pass


#~ line: 206, skipped: 6 ~~~~~~

@bind(ALvoid, [ALuint])
def alSourcePause(source): pass


#~ line: 215, skipped: 9 ~~~~~~

@bind(ALvoid, [ALuint])
def alSourceStop(source): pass


#~ line: 222, skipped: 7 ~~~~~~

@bind(ALvoid, [ALuint])
def alSourceRewind(source): pass


#~ line: 240, skipped: 18 ~~~~~~

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alGenBuffers(n, buffers): pass

@bind(ALvoid, [ALsizei, POINTER(ALuint)])
def alDeleteBuffers(n, buffers): pass

@bind(ALboolean, [ALuint])
def alIsBuffer(buffer): pass


#~ line: 251, skipped: 9 ~~~~~~

@bind(ALvoid, [ALuint, ALenum, POINTER(ALvoid), ALsizei, ALsizei])
def alBufferData(buffer, format, data, size, freq): pass


@bind(ALvoid, [ALuint, ALenum, POINTER(ALint)])
def alGetBufferi(buffer, param, value): pass

@bind(ALvoid, [ALuint, ALenum, POINTER(ALfloat)])
def alGetBufferf(buffer, param, value): pass


#~ line: 264, skipped: 9 ~~~~~~

@bind(ALvoid, [ALuint, ALsizei, POINTER(ALuint)])
def alSourceQueueBuffers(source, n, buffers): pass

@bind(ALvoid, [ALuint, ALsizei, POINTER(ALuint)])
def alSourceUnqueueBuffers(source, n, buffers): pass


#~ line: 270, skipped: 5 ~~~~~~

@bind(ALvoid, [ALenum])
def alDistanceModel(value): pass

@bind(ALvoid, [ALfloat])
def alDopplerFactor(value): pass

@bind(ALvoid, [ALfloat])
def alDopplerVelocity(value): pass



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/al.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


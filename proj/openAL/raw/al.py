#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/al.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AL_INVALID = (-1)
AL_ILLEGAL_ENUM = 0xA002 # = AL_INVALID_ENUM
AL_ILLEGAL_COMMAND = 0xA004 # = AL_INVALID_OPERATION

AL_VERSION_1_0 = None # empty
AL_VERSION_1_1 = None # empty

#~ line: 70, skipped: 4 ~~~~~~

# typedef ALboolean
ALboolean = c_byte

# typedef ALchar
ALchar = c_char

# typedef ALbyte
ALbyte = c_char

#~ line: 88, skipped: 12 ~~~~~~

# typedef ALint
ALint = c_int

# typedef ALuint
ALuint = c_uint

# typedef ALsizei
ALsizei = c_int

# typedef ALenum
ALenum = c_int

# typedef ALfloat
ALfloat = c_float

# typedef ALdouble
ALdouble = c_double

# typedef ALvoid
ALvoid = None

#~ line: 112, skipped: 6 ~~~~~~

AL_NONE = 0

AL_FALSE = 0

AL_TRUE = 1

AL_SOURCE_RELATIVE = 0x202

#~ line: 130, skipped: 9 ~~~~~~

AL_CONE_INNER_ANGLE = 0x1001

#~ line: 137, skipped: 7 ~~~~~~

AL_CONE_OUTER_ANGLE = 0x1002

#~ line: 145, skipped: 8 ~~~~~~

AL_PITCH = 0x1003

#~ line: 157, skipped: 12 ~~~~~~

AL_POSITION = 0x1004

AL_DIRECTION = 0x1005

AL_VELOCITY = 0x1006

#~ line: 171, skipped: 8 ~~~~~~

AL_LOOPING = 0x1007

#~ line: 178, skipped: 7 ~~~~~~

AL_BUFFER = 0x1009

#~ line: 191, skipped: 13 ~~~~~~

AL_GAIN = 0x100A

#~ line: 200, skipped: 9 ~~~~~~

AL_MIN_GAIN = 0x100D

#~ line: 209, skipped: 9 ~~~~~~

AL_MAX_GAIN = 0x100E

#~ line: 216, skipped: 7 ~~~~~~

AL_ORIENTATION = 0x100F

#~ line: 221, skipped: 5 ~~~~~~

AL_SOURCE_STATE = 0x1010
AL_INITIAL = 0x1011
AL_PLAYING = 0x1012
AL_PAUSED = 0x1013
AL_STOPPED = 0x1014

#~ line: 230, skipped: 5 ~~~~~~

AL_BUFFERS_QUEUED = 0x1015
AL_BUFFERS_PROCESSED = 0x1016

#~ line: 236, skipped: 5 ~~~~~~

AL_SEC_OFFSET = 0x1024
AL_SAMPLE_OFFSET = 0x1025
AL_BYTE_OFFSET = 0x1026

#~ line: 246, skipped: 8 ~~~~~~

AL_SOURCE_TYPE = 0x1027
AL_STATIC = 0x1028
AL_STREAMING = 0x1029
AL_UNDETERMINED = 0x1030

AL_FORMAT_MONO8 = 0x1100
AL_FORMAT_MONO16 = 0x1101
AL_FORMAT_STEREO8 = 0x1102
AL_FORMAT_STEREO16 = 0x1103

#~ line: 265, skipped: 10 ~~~~~~

AL_REFERENCE_DISTANCE = 0x1020

#~ line: 273, skipped: 8 ~~~~~~

AL_ROLLOFF_FACTOR = 0x1021

#~ line: 282, skipped: 9 ~~~~~~

AL_CONE_OUTER_GAIN = 0x1022

#~ line: 292, skipped: 10 ~~~~~~

AL_MAX_DISTANCE = 0x1023

#~ line: 300, skipped: 8 ~~~~~~

AL_FREQUENCY = 0x2001
AL_BITS = 0x2002
AL_CHANNELS = 0x2003
AL_SIZE = 0x2004

#~ line: 310, skipped: 7 ~~~~~~

AL_UNUSED = 0x2010
AL_PENDING = 0x2011
AL_PROCESSED = 0x2012

#~ line: 316, skipped: 4 ~~~~~~

AL_NO_ERROR = 0 # = AL_FALSE

#~ line: 321, skipped: 5 ~~~~~~

AL_INVALID_NAME = 0xA001

#~ line: 326, skipped: 5 ~~~~~~

AL_INVALID_ENUM = 0xA002

#~ line: 331, skipped: 5 ~~~~~~

AL_INVALID_VALUE = 0xA003

#~ line: 336, skipped: 5 ~~~~~~

AL_INVALID_OPERATION = 0xA004

#~ line: 342, skipped: 6 ~~~~~~

AL_OUT_OF_MEMORY = 0xA005

#~ line: 346, skipped: 4 ~~~~~~

AL_VENDOR = 0xB001
AL_VERSION = 0xB002
AL_RENDERER = 0xB003
AL_EXTENSIONS = 0xB004

#~ line: 356, skipped: 7 ~~~~~~

AL_DOPPLER_FACTOR = 0xC000

#~ line: 361, skipped: 5 ~~~~~~

AL_DOPPLER_VELOCITY = 0xC001

#~ line: 366, skipped: 5 ~~~~~~

AL_SPEED_OF_SOUND = 0xC003

#~ line: 375, skipped: 9 ~~~~~~

AL_DISTANCE_MODEL = 0xD000
AL_INVERSE_DISTANCE = 0xD001
AL_INVERSE_DISTANCE_CLAMPED = 0xD002
AL_LINEAR_DISTANCE = 0xD003
AL_LINEAR_DISTANCE_CLAMPED = 0xD004
AL_EXPONENT_DISTANCE = 0xD005
AL_EXPONENT_DISTANCE_CLAMPED = 0xD006

#~ line: 386, skipped: 5 ~~~~~~

@bind(None, [ALenum])
def alEnable(capability, _api_=None): 
    """alEnable(capability)
    
        capability : ALenum
    """
    return _api_(capability)
    

@bind(None, [ALenum])
def alDisable(capability, _api_=None): 
    """alDisable(capability)
    
        capability : ALenum
    """
    return _api_(capability)
    

@bind(ALboolean, [ALenum])
def alIsEnabled(capability, _api_=None): 
    """alIsEnabled(capability)
    
        capability : ALenum
    """
    return _api_(capability)
    

#~ line: 396, skipped: 6 ~~~~~~

@bind(c_char_p, [ALenum])
def alGetString(param, _api_=None): 
    """alGetString(param)
    
        param : ALenum
    """
    return _api_(param)
    

@bind(None, [ALenum, c_char_p])
def alGetBooleanv(param, data, _api_=None): 
    """alGetBooleanv(param, data)
    
        param : ALenum
        data : c_char_p
    """
    return _api_(param, data)
    

@bind(None, [ALenum, POINTER(c_int)])
def alGetIntegerv(param, data, _api_=None): 
    """alGetIntegerv(param, data)
    
        param : ALenum
        data : POINTER(c_int)
    """
    return _api_(param, data)
    

@bind(None, [ALenum, POINTER(c_float)])
def alGetFloatv(param, data, _api_=None): 
    """alGetFloatv(param, data)
    
        param : ALenum
        data : POINTER(c_float)
    """
    return _api_(param, data)
    

@bind(None, [ALenum, POINTER(c_double)])
def alGetDoublev(param, data, _api_=None): 
    """alGetDoublev(param, data)
    
        param : ALenum
        data : POINTER(c_double)
    """
    return _api_(param, data)
    

@bind(ALboolean, [ALenum])
def alGetBoolean(param, _api_=None): 
    """alGetBoolean(param)
    
        param : ALenum
    """
    return _api_(param)
    

@bind(ALint, [ALenum])
def alGetInteger(param, _api_=None): 
    """alGetInteger(param)
    
        param : ALenum
    """
    return _api_(param)
    

@bind(ALfloat, [ALenum])
def alGetFloat(param, _api_=None): 
    """alGetFloat(param)
    
        param : ALenum
    """
    return _api_(param)
    

@bind(ALdouble, [ALenum])
def alGetDouble(param, _api_=None): 
    """alGetDouble(param)
    
        param : ALenum
    """
    return _api_(param)
    

#~ line: 419, skipped: 7 ~~~~~~

@bind(ALenum, [])
def alGetError(_api_=None): 
    """alGetError()
    
        
    """
    return _api_()
    

#~ line: 427, skipped: 8 ~~~~~~

@bind(ALboolean, [c_char_p])
def alIsExtensionPresent(extname, _api_=None): 
    """alIsExtensionPresent(extname)
    
        extname : c_char_p
    """
    return _api_(extname)
    

@bind(c_void_p, [c_char_p])
def alGetProcAddress(fname, _api_=None): 
    """alGetProcAddress(fname)
    
        fname : c_char_p
    """
    return _api_(fname)
    

@bind(ALenum, [c_char_p])
def alGetEnumValue(ename, _api_=None): 
    """alGetEnumValue(ename)
    
        ename : c_char_p
    """
    return _api_(ename)
    

#~ line: 450, skipped: 19 ~~~~~~

@bind(None, [ALenum, ALfloat])
def alListenerf(param, value, _api_=None): 
    """alListenerf(param, value)
    
        param : ALenum
        value : ALfloat
    """
    return _api_(param, value)
    

@bind(None, [ALenum, ALfloat, ALfloat, ALfloat])
def alListener3f(param, value1, value2, value3, _api_=None): 
    """alListener3f(param, value1, value2, value3)
    
        param : ALenum
        value1 : ALfloat
        value2 : ALfloat
        value3 : ALfloat
    """
    return _api_(param, value1, value2, value3)
    

@bind(None, [ALenum, POINTER(c_float)])
def alListenerfv(param, values, _api_=None): 
    """alListenerfv(param, values)
    
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(param, values)
    

@bind(None, [ALenum, ALint])
def alListeneri(param, value, _api_=None): 
    """alListeneri(param, value)
    
        param : ALenum
        value : ALint
    """
    return _api_(param, value)
    

@bind(None, [ALenum, ALint, ALint, ALint])
def alListener3i(param, value1, value2, value3, _api_=None): 
    """alListener3i(param, value1, value2, value3)
    
        param : ALenum
        value1 : ALint
        value2 : ALint
        value3 : ALint
    """
    return _api_(param, value1, value2, value3)
    

@bind(None, [ALenum, POINTER(c_int)])
def alListeneriv(param, values, _api_=None): 
    """alListeneriv(param, values)
    
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(param, values)
    

#~ line: 465, skipped: 5 ~~~~~~

@bind(None, [ALenum, POINTER(c_float)])
def alGetListenerf(param, value, _api_=None): 
    """alGetListenerf(param, value)
    
        param : ALenum
        value : POINTER(c_float)
    """
    return _api_(param, value)
    

@bind(None, [ALenum, POINTER(c_float), POINTER(c_float), POINTER(c_float)])
def alGetListener3f(param, value1, value2, value3, _api_=None): 
    """alGetListener3f(param, value1, value2, value3)
    
        param : ALenum
        value1 : POINTER(c_float)
        value2 : POINTER(c_float)
        value3 : POINTER(c_float)
    """
    return _api_(param, value1, value2, value3)
    

@bind(None, [ALenum, POINTER(c_float)])
def alGetListenerfv(param, values, _api_=None): 
    """alGetListenerfv(param, values)
    
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(param, values)
    

@bind(None, [ALenum, POINTER(c_int)])
def alGetListeneri(param, value, _api_=None): 
    """alGetListeneri(param, value)
    
        param : ALenum
        value : POINTER(c_int)
    """
    return _api_(param, value)
    

@bind(None, [ALenum, POINTER(c_int), POINTER(c_int), POINTER(c_int)])
def alGetListener3i(param, value1, value2, value3, _api_=None): 
    """alGetListener3i(param, value1, value2, value3)
    
        param : ALenum
        value1 : POINTER(c_int)
        value2 : POINTER(c_int)
        value3 : POINTER(c_int)
    """
    return _api_(param, value1, value2, value3)
    

@bind(None, [ALenum, POINTER(c_int)])
def alGetListeneriv(param, values, _api_=None): 
    """alGetListeneriv(param, values)
    
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(param, values)
    

#~ line: 512, skipped: 37 ~~~~~~

@bind(None, [ALsizei, POINTER(c_uint)])
def alGenSources(n, sources, _api_=None): 
    """alGenSources(n, sources)
    
        n : ALsizei
        sources : POINTER(c_uint)
    """
    return _api_(n, sources)
    

@bind(None, [ALsizei, POINTER(c_uint)])
def alDeleteSources(n, sources, _api_=None): 
    """alDeleteSources(n, sources)
    
        n : ALsizei
        sources : POINTER(c_uint)
    """
    return _api_(n, sources)
    

@bind(ALboolean, [ALuint])
def alIsSource(sid, _api_=None): 
    """alIsSource(sid)
    
        sid : ALuint
    """
    return _api_(sid)
    

#~ line: 523, skipped: 5 ~~~~~~

@bind(None, [ALuint, ALenum, ALfloat])
def alSourcef(sid, param, value, _api_=None): 
    """alSourcef(sid, param, value)
    
        sid : ALuint
        param : ALenum
        value : ALfloat
    """
    return _api_(sid, param, value)
    

@bind(None, [ALuint, ALenum, ALfloat, ALfloat, ALfloat])
def alSource3f(sid, param, value1, value2, value3, _api_=None): 
    """alSource3f(sid, param, value1, value2, value3)
    
        sid : ALuint
        param : ALenum
        value1 : ALfloat
        value2 : ALfloat
        value3 : ALfloat
    """
    return _api_(sid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alSourcefv(sid, param, values, _api_=None): 
    """alSourcefv(sid, param, values)
    
        sid : ALuint
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(sid, param, values)
    

@bind(None, [ALuint, ALenum, ALint])
def alSourcei(sid, param, value, _api_=None): 
    """alSourcei(sid, param, value)
    
        sid : ALuint
        param : ALenum
        value : ALint
    """
    return _api_(sid, param, value)
    

@bind(None, [ALuint, ALenum, ALint, ALint, ALint])
def alSource3i(sid, param, value1, value2, value3, _api_=None): 
    """alSource3i(sid, param, value1, value2, value3)
    
        sid : ALuint
        param : ALenum
        value1 : ALint
        value2 : ALint
        value3 : ALint
    """
    return _api_(sid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alSourceiv(sid, param, values, _api_=None): 
    """alSourceiv(sid, param, values)
    
        sid : ALuint
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(sid, param, values)
    

#~ line: 538, skipped: 5 ~~~~~~

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alGetSourcef(sid, param, value, _api_=None): 
    """alGetSourcef(sid, param, value)
    
        sid : ALuint
        param : ALenum
        value : POINTER(c_float)
    """
    return _api_(sid, param, value)
    

@bind(None, [ALuint, ALenum, POINTER(c_float), POINTER(c_float), POINTER(c_float)])
def alGetSource3f(sid, param, value1, value2, value3, _api_=None): 
    """alGetSource3f(sid, param, value1, value2, value3)
    
        sid : ALuint
        param : ALenum
        value1 : POINTER(c_float)
        value2 : POINTER(c_float)
        value3 : POINTER(c_float)
    """
    return _api_(sid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alGetSourcefv(sid, param, values, _api_=None): 
    """alGetSourcefv(sid, param, values)
    
        sid : ALuint
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(sid, param, values)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alGetSourcei(sid, param, value, _api_=None): 
    """alGetSourcei(sid, param, value)
    
        sid : ALuint
        param : ALenum
        value : POINTER(c_int)
    """
    return _api_(sid, param, value)
    

@bind(None, [ALuint, ALenum, POINTER(c_int), POINTER(c_int), POINTER(c_int)])
def alGetSource3i(sid, param, value1, value2, value3, _api_=None): 
    """alGetSource3i(sid, param, value1, value2, value3)
    
        sid : ALuint
        param : ALenum
        value1 : POINTER(c_int)
        value2 : POINTER(c_int)
        value3 : POINTER(c_int)
    """
    return _api_(sid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alGetSourceiv(sid, param, values, _api_=None): 
    """alGetSourceiv(sid, param, values)
    
        sid : ALuint
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(sid, param, values)
    

#~ line: 556, skipped: 8 ~~~~~~

@bind(None, [ALsizei, POINTER(c_uint)])
def alSourcePlayv(ns, sids, _api_=None): 
    """alSourcePlayv(ns, sids)
    
        ns : ALsizei
        sids : POINTER(c_uint)
    """
    return _api_(ns, sids)
    

@bind(None, [ALsizei, POINTER(c_uint)])
def alSourceStopv(ns, sids, _api_=None): 
    """alSourceStopv(ns, sids)
    
        ns : ALsizei
        sids : POINTER(c_uint)
    """
    return _api_(ns, sids)
    

@bind(None, [ALsizei, POINTER(c_uint)])
def alSourceRewindv(ns, sids, _api_=None): 
    """alSourceRewindv(ns, sids)
    
        ns : ALsizei
        sids : POINTER(c_uint)
    """
    return _api_(ns, sids)
    

@bind(None, [ALsizei, POINTER(c_uint)])
def alSourcePausev(ns, sids, _api_=None): 
    """alSourcePausev(ns, sids)
    
        ns : ALsizei
        sids : POINTER(c_uint)
    """
    return _api_(ns, sids)
    

#~ line: 572, skipped: 7 ~~~~~~

@bind(None, [ALuint])
def alSourcePlay(sid, _api_=None): 
    """alSourcePlay(sid)
    
        sid : ALuint
    """
    return _api_(sid)
    

@bind(None, [ALuint])
def alSourceStop(sid, _api_=None): 
    """alSourceStop(sid)
    
        sid : ALuint
    """
    return _api_(sid)
    

@bind(None, [ALuint])
def alSourceRewind(sid, _api_=None): 
    """alSourceRewind(sid)
    
        sid : ALuint
    """
    return _api_(sid)
    

@bind(None, [ALuint])
def alSourcePause(sid, _api_=None): 
    """alSourcePause(sid)
    
        sid : ALuint
    """
    return _api_(sid)
    

#~ line: 586, skipped: 5 ~~~~~~

@bind(None, [ALuint, ALsizei, POINTER(c_uint)])
def alSourceQueueBuffers(sid, numEntries, bids, _api_=None): 
    """alSourceQueueBuffers(sid, numEntries, bids)
    
        sid : ALuint
        numEntries : ALsizei
        bids : POINTER(c_uint)
    """
    return _api_(sid, numEntries, bids)
    

@bind(None, [ALuint, ALsizei, POINTER(c_uint)])
def alSourceUnqueueBuffers(sid, numEntries, bids, _api_=None): 
    """alSourceUnqueueBuffers(sid, numEntries, bids)
    
        sid : ALuint
        numEntries : ALsizei
        bids : POINTER(c_uint)
    """
    return _api_(sid, numEntries, bids)
    

#~ line: 606, skipped: 18 ~~~~~~

@bind(None, [ALsizei, POINTER(c_uint)])
def alGenBuffers(n, buffers, _api_=None): 
    """alGenBuffers(n, buffers)
    
        n : ALsizei
        buffers : POINTER(c_uint)
    """
    return _api_(n, buffers)
    

@bind(None, [ALsizei, POINTER(c_uint)])
def alDeleteBuffers(n, buffers, _api_=None): 
    """alDeleteBuffers(n, buffers)
    
        n : ALsizei
        buffers : POINTER(c_uint)
    """
    return _api_(n, buffers)
    

@bind(ALboolean, [ALuint])
def alIsBuffer(bid, _api_=None): 
    """alIsBuffer(bid)
    
        bid : ALuint
    """
    return _api_(bid)
    

@bind(None, [ALuint, ALenum, c_void_p, ALsizei, ALsizei])
def alBufferData(bid, format, data, size, freq, _api_=None): 
    """alBufferData(bid, format, data, size, freq)
    
        bid : ALuint
        format : ALenum
        data : c_void_p
        size : ALsizei
        freq : ALsizei
    """
    return _api_(bid, format, data, size, freq)
    

#~ line: 620, skipped: 5 ~~~~~~

@bind(None, [ALuint, ALenum, ALfloat])
def alBufferf(bid, param, value, _api_=None): 
    """alBufferf(bid, param, value)
    
        bid : ALuint
        param : ALenum
        value : ALfloat
    """
    return _api_(bid, param, value)
    

@bind(None, [ALuint, ALenum, ALfloat, ALfloat, ALfloat])
def alBuffer3f(bid, param, value1, value2, value3, _api_=None): 
    """alBuffer3f(bid, param, value1, value2, value3)
    
        bid : ALuint
        param : ALenum
        value1 : ALfloat
        value2 : ALfloat
        value3 : ALfloat
    """
    return _api_(bid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alBufferfv(bid, param, values, _api_=None): 
    """alBufferfv(bid, param, values)
    
        bid : ALuint
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(bid, param, values)
    

@bind(None, [ALuint, ALenum, ALint])
def alBufferi(bid, param, value, _api_=None): 
    """alBufferi(bid, param, value)
    
        bid : ALuint
        param : ALenum
        value : ALint
    """
    return _api_(bid, param, value)
    

@bind(None, [ALuint, ALenum, ALint, ALint, ALint])
def alBuffer3i(bid, param, value1, value2, value3, _api_=None): 
    """alBuffer3i(bid, param, value1, value2, value3)
    
        bid : ALuint
        param : ALenum
        value1 : ALint
        value2 : ALint
        value3 : ALint
    """
    return _api_(bid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alBufferiv(bid, param, values, _api_=None): 
    """alBufferiv(bid, param, values)
    
        bid : ALuint
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(bid, param, values)
    

#~ line: 635, skipped: 5 ~~~~~~

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alGetBufferf(bid, param, value, _api_=None): 
    """alGetBufferf(bid, param, value)
    
        bid : ALuint
        param : ALenum
        value : POINTER(c_float)
    """
    return _api_(bid, param, value)
    

@bind(None, [ALuint, ALenum, POINTER(c_float), POINTER(c_float), POINTER(c_float)])
def alGetBuffer3f(bid, param, value1, value2, value3, _api_=None): 
    """alGetBuffer3f(bid, param, value1, value2, value3)
    
        bid : ALuint
        param : ALenum
        value1 : POINTER(c_float)
        value2 : POINTER(c_float)
        value3 : POINTER(c_float)
    """
    return _api_(bid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_float)])
def alGetBufferfv(bid, param, values, _api_=None): 
    """alGetBufferfv(bid, param, values)
    
        bid : ALuint
        param : ALenum
        values : POINTER(c_float)
    """
    return _api_(bid, param, values)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alGetBufferi(bid, param, value, _api_=None): 
    """alGetBufferi(bid, param, value)
    
        bid : ALuint
        param : ALenum
        value : POINTER(c_int)
    """
    return _api_(bid, param, value)
    

@bind(None, [ALuint, ALenum, POINTER(c_int), POINTER(c_int), POINTER(c_int)])
def alGetBuffer3i(bid, param, value1, value2, value3, _api_=None): 
    """alGetBuffer3i(bid, param, value1, value2, value3)
    
        bid : ALuint
        param : ALenum
        value1 : POINTER(c_int)
        value2 : POINTER(c_int)
        value3 : POINTER(c_int)
    """
    return _api_(bid, param, value1, value2, value3)
    

@bind(None, [ALuint, ALenum, POINTER(c_int)])
def alGetBufferiv(bid, param, values, _api_=None): 
    """alGetBufferiv(bid, param, values)
    
        bid : ALuint
        param : ALenum
        values : POINTER(c_int)
    """
    return _api_(bid, param, values)
    

#~ line: 651, skipped: 6 ~~~~~~

@bind(None, [ALfloat])
def alDopplerFactor(value, _api_=None): 
    """alDopplerFactor(value)
    
        value : ALfloat
    """
    return _api_(value)
    

@bind(None, [ALfloat])
def alDopplerVelocity(value, _api_=None): 
    """alDopplerVelocity(value)
    
        value : ALfloat
    """
    return _api_(value)
    

@bind(None, [ALfloat])
def alSpeedOfSound(value, _api_=None): 
    """alSpeedOfSound(value)
    
        value : ALfloat
    """
    return _api_(value)
    

@bind(None, [ALenum])
def alDistanceModel(distanceModel, _api_=None): 
    """alDistanceModel(distanceModel)
    
        distanceModel : ALenum
    """
    return _api_(distanceModel)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/al.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


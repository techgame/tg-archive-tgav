#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *
from al import *
from alc import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/alut.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALUT_API_MAJOR_VERSION = 1
ALUT_API_MINOR_VERSION = 1

ALUT_ERROR_NO_ERROR = 0
ALUT_ERROR_OUT_OF_MEMORY = 0x200
ALUT_ERROR_INVALID_ENUM = 0x201
ALUT_ERROR_INVALID_VALUE = 0x202
ALUT_ERROR_INVALID_OPERATION = 0x203
ALUT_ERROR_NO_CURRENT_CONTEXT = 0x204
ALUT_ERROR_AL_ERROR_ON_ENTRY = 0x205
ALUT_ERROR_ALC_ERROR_ON_ENTRY = 0x206
ALUT_ERROR_OPEN_DEVICE = 0x207
ALUT_ERROR_CLOSE_DEVICE = 0x208
ALUT_ERROR_CREATE_CONTEXT = 0x209
ALUT_ERROR_MAKE_CONTEXT_CURRENT = 0x20A
ALUT_ERROR_DESTROY_CONTEXT = 0x20B
ALUT_ERROR_GEN_BUFFERS = 0x20C
ALUT_ERROR_BUFFER_DATA = 0x20D
ALUT_ERROR_IO_ERROR = 0x20E
ALUT_ERROR_UNSUPPORTED_FILE_TYPE = 0x20F
ALUT_ERROR_UNSUPPORTED_FILE_SUBTYPE = 0x210
ALUT_ERROR_CORRUPT_OR_TRUNCATED_DATA = 0x211

ALUT_WAVEFORM_SINE = 0x100
ALUT_WAVEFORM_SQUARE = 0x101
ALUT_WAVEFORM_SAWTOOTH = 0x102
ALUT_WAVEFORM_WHITENOISE = 0x103
ALUT_WAVEFORM_IMPULSE = 0x104

ALUT_LOADER_BUFFER = 0x300
ALUT_LOADER_MEMORY = 0x301

@bind(ALboolean, [POINTER(c_int), c_char_p])
def alutInit(argcp, argv, _api_=None): 
    """alutInit(argcp, argv)
    
        argcp : POINTER(c_int)
        argv : c_char_p
    """
    return _api_(argcp, argv)
    
@bind(ALboolean, [POINTER(c_int), c_char_p])
def alutInitWithoutContext(argcp, argv, _api_=None): 
    """alutInitWithoutContext(argcp, argv)
    
        argcp : POINTER(c_int)
        argv : c_char_p
    """
    return _api_(argcp, argv)
    
@bind(ALboolean, [])
def alutExit(_api_=None): 
    """alutExit()
    
        
    """
    return _api_()
    

@bind(ALenum, [])
def alutGetError(_api_=None): 
    """alutGetError()
    
        
    """
    return _api_()
    
@bind(c_char_p, [ALenum])
def alutGetErrorString(error, _api_=None): 
    """alutGetErrorString(error)
    
        error : ALenum
    """
    return _api_(error)
    

@bind(ALuint, [c_char_p])
def alutCreateBufferFromFile(fileName, _api_=None): 
    """alutCreateBufferFromFile(fileName)
    
        fileName : c_char_p
    """
    return _api_(fileName)
    
@bind(ALuint, [c_void_p, ALsizei])
def alutCreateBufferFromFileImage(data, length, _api_=None): 
    """alutCreateBufferFromFileImage(data, length)
    
        data : c_void_p
        length : ALsizei
    """
    return _api_(data, length)
    
@bind(ALuint, [])
def alutCreateBufferHelloWorld(_api_=None): 
    """alutCreateBufferHelloWorld()
    
        
    """
    return _api_()
    
@bind(ALuint, [ALenum, ALfloat, ALfloat, ALfloat])
def alutCreateBufferWaveform(waveshape, frequency, phase, duration, _api_=None): 
    """alutCreateBufferWaveform(waveshape, frequency, phase, duration)
    
        waveshape : ALenum
        frequency : ALfloat
        phase : ALfloat
        duration : ALfloat
    """
    return _api_(waveshape, frequency, phase, duration)
    

@bind(c_void_p, [c_char_p, POINTER(c_int), POINTER(c_int), POINTER(c_float)])
def alutLoadMemoryFromFile(fileName, format, size, frequency, _api_=None): 
    """alutLoadMemoryFromFile(fileName, format, size, frequency)
    
        fileName : c_char_p
        format : POINTER(c_int)
        size : POINTER(c_int)
        frequency : POINTER(c_float)
    """
    return _api_(fileName, format, size, frequency)
    
@bind(c_void_p, [c_void_p, ALsizei, POINTER(c_int), POINTER(c_int), POINTER(c_float)])
def alutLoadMemoryFromFileImage(data, length, format, size, frequency, _api_=None): 
    """alutLoadMemoryFromFileImage(data, length, format, size, frequency)
    
        data : c_void_p
        length : ALsizei
        format : POINTER(c_int)
        size : POINTER(c_int)
        frequency : POINTER(c_float)
    """
    return _api_(data, length, format, size, frequency)
    
@bind(c_void_p, [POINTER(c_int), POINTER(c_int), POINTER(c_float)])
def alutLoadMemoryHelloWorld(format, size, frequency, _api_=None): 
    """alutLoadMemoryHelloWorld(format, size, frequency)
    
        format : POINTER(c_int)
        size : POINTER(c_int)
        frequency : POINTER(c_float)
    """
    return _api_(format, size, frequency)
    
@bind(c_void_p, [ALenum, ALfloat, ALfloat, ALfloat, POINTER(c_int), POINTER(c_int), POINTER(c_float)])
def alutLoadMemoryWaveform(waveshape, frequency, phase, duration, format, size, freq, _api_=None): 
    """alutLoadMemoryWaveform(waveshape, frequency, phase, duration, format, size, freq)
    
        waveshape : ALenum
        frequency : ALfloat
        phase : ALfloat
        duration : ALfloat
        format : POINTER(c_int)
        size : POINTER(c_int)
        freq : POINTER(c_float)
    """
    return _api_(waveshape, frequency, phase, duration, format, size, freq)
    

@bind(c_char_p, [ALenum])
def alutGetMIMETypes(loader, _api_=None): 
    """alutGetMIMETypes(loader)
    
        loader : ALenum
    """
    return _api_(loader)
    

@bind(ALint, [])
def alutGetMajorVersion(_api_=None): 
    """alutGetMajorVersion()
    
        
    """
    return _api_()
    
@bind(ALint, [])
def alutGetMinorVersion(_api_=None): 
    """alutGetMinorVersion()
    
        
    """
    return _api_()
    

@bind(ALboolean, [ALfloat])
def alutSleep(duration, _api_=None): 
    """alutSleep(duration)
    
        duration : ALfloat
    """
    return _api_(duration)
    

#~ line: 110, skipped: 4 ~~~~~~

@bind(None, [c_char_p, POINTER(c_int), c_void_p, POINTER(c_int), POINTER(c_int)])
def alutLoadWAVFile(fileName, format, data, size, frequency, _api_=None): 
    """alutLoadWAVFile(fileName, format, data, size, frequency)
    
        fileName : c_char_p
        format : POINTER(c_int)
        data : c_void_p
        size : POINTER(c_int)
        frequency : POINTER(c_int)
    """
    return _api_(fileName, format, data, size, frequency)
    
@bind(None, [c_char_p, POINTER(c_int), c_void_p, POINTER(c_int), POINTER(c_int)])
def alutLoadWAVMemory(buffer, format, data, size, frequency, _api_=None): 
    """alutLoadWAVMemory(buffer, format, data, size, frequency)
    
        buffer : c_char_p
        format : POINTER(c_int)
        data : c_void_p
        size : POINTER(c_int)
        frequency : POINTER(c_int)
    """
    return _api_(buffer, format, data, size, frequency)
    

#~ line: 116, skipped: 5 ~~~~~~

@bind(None, [ALenum, c_void_p, ALsizei, ALsizei])
def alutUnloadWAV(format, data, size, frequency, _api_=None): 
    """alutUnloadWAV(format, data, size, frequency)
    
        format : ALenum
        data : c_void_p
        size : ALsizei
        frequency : ALsizei
    """
    return _api_(format, data, size, frequency)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/alut.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


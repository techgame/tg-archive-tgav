#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/alc.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALC_INVALID = 0

ALC_VERSION_0_1 = 1

#ALCdevice_struct = c_void_p # Structure with empty _fields_
# typedef ALCdevice
#ALCdevice = ALCdevice_struct
ALCdevice_p = c_void_p
#ALCcontext_struct = c_void_p # Structure with empty _fields_
# typedef ALCcontext
#ALCcontext = ALCcontext_struct
ALCcontext_p = c_void_p

#~ line: 49, skipped: 4 ~~~~~~

# typedef ALCboolean
ALCboolean = c_byte

# typedef ALCchar
ALCchar = c_char

#~ line: 67, skipped: 15 ~~~~~~

# typedef ALCint
ALCint = c_int

# typedef ALCuint
ALCuint = c_uint

# typedef ALCsizei
ALCsizei = c_int

# typedef ALCenum
ALCenum = c_int

#~ line: 85, skipped: 9 ~~~~~~

# typedef ALCvoid
ALCvoid = None

#~ line: 91, skipped: 6 ~~~~~~

ALC_FALSE = 0

ALC_TRUE = 1

#~ line: 99, skipped: 5 ~~~~~~

ALC_FREQUENCY = 0x1007

#~ line: 104, skipped: 5 ~~~~~~

ALC_REFRESH = 0x1008

#~ line: 109, skipped: 5 ~~~~~~

ALC_SYNC = 0x1009

#~ line: 114, skipped: 5 ~~~~~~

ALC_MONO_SOURCES = 0x1010

#~ line: 119, skipped: 5 ~~~~~~

ALC_STEREO_SOURCES = 0x1011

#~ line: 128, skipped: 9 ~~~~~~

ALC_NO_ERROR = 0 # = ALC_FALSE

#~ line: 133, skipped: 5 ~~~~~~

ALC_INVALID_DEVICE = 0xA001

#~ line: 138, skipped: 5 ~~~~~~

ALC_INVALID_CONTEXT = 0xA002

#~ line: 143, skipped: 5 ~~~~~~

ALC_INVALID_ENUM = 0xA003

#~ line: 148, skipped: 5 ~~~~~~

ALC_INVALID_VALUE = 0xA004

#~ line: 153, skipped: 5 ~~~~~~

ALC_OUT_OF_MEMORY = 0xA005

#~ line: 159, skipped: 6 ~~~~~~

ALC_DEFAULT_DEVICE_SPECIFIER = 0x1004
ALC_DEVICE_SPECIFIER = 0x1005
ALC_EXTENSIONS = 0x1006

ALC_MAJOR_VERSION = 0x1000
ALC_MINOR_VERSION = 0x1001

ALC_ATTRIBUTES_SIZE = 0x1002
ALC_ALL_ATTRIBUTES = 0x1003

#~ line: 172, skipped: 5 ~~~~~~

ALC_CAPTURE_DEVICE_SPECIFIER = 0x310
ALC_CAPTURE_DEFAULT_DEVICE_SPECIFIER = 0x311
ALC_CAPTURE_SAMPLES = 0x312

#~ line: 180, skipped: 6 ~~~~~~

@bind(ALCcontext_p, [ALCdevice_p, POINTER(c_int)])
def alcCreateContext(device, attrlist, _api_=None): 
    """alcCreateContext(device, attrlist)
    
        device : ALCdevice_p
        attrlist : POINTER(c_int)
    """
    return _api_(device, attrlist)
    

@bind(ALCboolean, [ALCcontext_p])
def alcMakeContextCurrent(context, _api_=None): 
    """alcMakeContextCurrent(context)
    
        context : ALCcontext_p
    """
    return _api_(context)
    

@bind(None, [ALCcontext_p])
def alcProcessContext(context, _api_=None): 
    """alcProcessContext(context)
    
        context : ALCcontext_p
    """
    return _api_(context)
    

@bind(None, [ALCcontext_p])
def alcSuspendContext(context, _api_=None): 
    """alcSuspendContext(context)
    
        context : ALCcontext_p
    """
    return _api_(context)
    

@bind(None, [ALCcontext_p])
def alcDestroyContext(context, _api_=None): 
    """alcDestroyContext(context)
    
        context : ALCcontext_p
    """
    return _api_(context)
    

@bind(ALCcontext_p, [])
def alcGetCurrentContext(_api_=None): 
    """alcGetCurrentContext()
    
        
    """
    return _api_()
    

@bind(ALCdevice_p, [ALCcontext_p])
def alcGetContextsDevice(context, _api_=None): 
    """alcGetContextsDevice(context)
    
        context : ALCcontext_p
    """
    return _api_(context)
    

#~ line: 198, skipped: 6 ~~~~~~

@bind(ALCdevice_p, [c_char_p])
def alcOpenDevice(devicename, _api_=None): 
    """alcOpenDevice(devicename)
    
        devicename : c_char_p
    """
    return _api_(devicename)
    

@bind(ALCboolean, [ALCdevice_p])
def alcCloseDevice(device, _api_=None): 
    """alcCloseDevice(device)
    
        device : ALCdevice_p
    """
    return _api_(device)
    

#~ line: 207, skipped: 7 ~~~~~~

@bind(ALCenum, [ALCdevice_p])
def alcGetError(device, _api_=None): 
    """alcGetError(device)
    
        device : ALCdevice_p
    """
    return _api_(device)
    

#~ line: 215, skipped: 8 ~~~~~~

@bind(ALCboolean, [ALCdevice_p, c_char_p])
def alcIsExtensionPresent(device, extname, _api_=None): 
    """alcIsExtensionPresent(device, extname)
    
        device : ALCdevice_p
        extname : c_char_p
    """
    return _api_(device, extname)
    

@bind(c_void_p, [ALCdevice_p, c_char_p])
def alcGetProcAddress(device, funcname, _api_=None): 
    """alcGetProcAddress(device, funcname)
    
        device : ALCdevice_p
        funcname : c_char_p
    """
    return _api_(device, funcname)
    

@bind(ALCenum, [ALCdevice_p, c_char_p])
def alcGetEnumValue(device, enumname, _api_=None): 
    """alcGetEnumValue(device, enumname)
    
        device : ALCdevice_p
        enumname : c_char_p
    """
    return _api_(device, enumname)
    

#~ line: 225, skipped: 6 ~~~~~~

@bind(POINTER(c_char), [ALCdevice_p, ALCenum])
def alcGetString(device, param, _api_=None): 
    """alcGetString(device, param)
    
        device : ALCdevice_p
        param : ALCenum
    """
    return _api_(device, param)
    

@bind(None, [ALCdevice_p, ALCenum, ALCsizei, POINTER(c_int)])
def alcGetIntegerv(device, param, size, data, _api_=None): 
    """alcGetIntegerv(device, param, size, data)
    
        device : ALCdevice_p
        param : ALCenum
        size : ALCsizei
        data : POINTER(c_int)
    """
    return _api_(device, param, size, data)
    

#~ line: 233, skipped: 6 ~~~~~~

@bind(ALCdevice_p, [c_char_p, ALCuint, ALCenum, ALCsizei])
def alcCaptureOpenDevice(devicename, frequency, format, buffersize, _api_=None): 
    """alcCaptureOpenDevice(devicename, frequency, format, buffersize)
    
        devicename : c_char_p
        frequency : ALCuint
        format : ALCenum
        buffersize : ALCsizei
    """
    return _api_(devicename, frequency, format, buffersize)
    

@bind(ALCboolean, [ALCdevice_p])
def alcCaptureCloseDevice(device, _api_=None): 
    """alcCaptureCloseDevice(device)
    
        device : ALCdevice_p
    """
    return _api_(device)
    

@bind(None, [ALCdevice_p])
def alcCaptureStart(device, _api_=None): 
    """alcCaptureStart(device)
    
        device : ALCdevice_p
    """
    return _api_(device)
    

@bind(None, [ALCdevice_p])
def alcCaptureStop(device, _api_=None): 
    """alcCaptureStop(device)
    
        device : ALCdevice_p
    """
    return _api_(device)
    

@bind(None, [ALCdevice_p, c_void_p, ALCsizei])
def alcCaptureSamples(device, buffer, samples, _api_=None): 
    """alcCaptureSamples(device, buffer, samples)
    
        device : ALCdevice_p
        buffer : c_void_p
        samples : ALCsizei
    """
    return _api_(device, buffer, samples)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/alc.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


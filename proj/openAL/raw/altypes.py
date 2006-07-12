#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/altypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALboolean(c_byte):
    """typedef ALboolean"""

class ALbyte(c_char):
    """typedef ALbyte"""

class ALubyte(c_ubyte):
    """typedef ALubyte"""

#~ line: 45, skipped: 9 ~~~~~~

class ALuint(c_uint):
    """typedef ALuint"""
    def __int__(self):
        return self.value

class ALint(c_int):
    """typedef ALint"""
    def __int__(self):
        return self.value

class ALfloat(c_float):
    """typedef ALfloat"""
    def __float__(self):
        return self.value

class ALdouble(c_double):
    """typedef ALdouble"""
    def __float__(self):
        return self.value

class ALsizei(c_uint):
    """typedef ALsizei"""
    def __int__(self):
        return self.value
    def __index__(self):
        return self.value

ALvoid = None # typedef ALvoid

class ALenum(c_int):
    """typedef ALenum"""
    def __int__(self):
        return self.value

AL_INVALID = (-1)

AL_NONE = 0

AL_FALSE = 0

AL_TRUE = 1

#~ line: 81, skipped: 6 ~~~~~~

AL_SOURCE_TYPE = 0x200

AL_SOURCE_ABSOLUTE = 0x201

AL_SOURCE_RELATIVE = 0x202

#~ line: 94, skipped: 7 ~~~~~~

AL_CONE_INNER_ANGLE = 0x1001

#~ line: 101, skipped: 7 ~~~~~~

AL_CONE_OUTER_ANGLE = 0x1002

#~ line: 109, skipped: 8 ~~~~~~

AL_PITCH = 0x1003

#~ line: 121, skipped: 12 ~~~~~~

AL_POSITION = 0x1004

AL_DIRECTION = 0x1005

AL_VELOCITY = 0x1006

#~ line: 135, skipped: 8 ~~~~~~

AL_LOOPING = 0x1007

#~ line: 142, skipped: 7 ~~~~~~

AL_BUFFER = 0x1009

#~ line: 155, skipped: 13 ~~~~~~

AL_GAIN = 0x100A

#~ line: 162, skipped: 7 ~~~~~~

AL_MIN_GAIN = 0x100D

#~ line: 169, skipped: 7 ~~~~~~

AL_MAX_GAIN = 0x100E

#~ line: 176, skipped: 7 ~~~~~~

AL_ORIENTATION = 0x100F

#~ line: 185, skipped: 9 ~~~~~~

AL_REFERENCE_DISTANCE = 0x1020

#~ line: 193, skipped: 8 ~~~~~~

AL_ROLLOFF_FACTOR = 0x1021

#~ line: 206, skipped: 13 ~~~~~~

AL_CONE_OUTER_GAIN = 0x1022

#~ line: 213, skipped: 7 ~~~~~~

AL_MAX_DISTANCE = 0x1023

#~ line: 218, skipped: 5 ~~~~~~

AL_SOURCE_STATE = 0x1010
AL_INITIAL = 0x1011
AL_PLAYING = 0x1012
AL_PAUSED = 0x1013
AL_STOPPED = 0x1014

#~ line: 227, skipped: 5 ~~~~~~

AL_BUFFERS_QUEUED = 0x1015
AL_BUFFERS_PROCESSED = 0x1016

AL_FORMAT_MONO8 = 0x1100
AL_FORMAT_MONO16 = 0x1101
AL_FORMAT_STEREO8 = 0x1102
AL_FORMAT_STEREO16 = 0x1103

#~ line: 242, skipped: 8 ~~~~~~

AL_FREQUENCY = 0x2001
AL_BITS = 0x2002
AL_CHANNELS = 0x2003
AL_SIZE = 0x2004
AL_DATA = 0x2005

#~ line: 253, skipped: 7 ~~~~~~

AL_UNUSED = 0x2010
AL_PENDING = 0x2011
AL_PROCESSED = 0x2012

AL_NO_ERROR = 0 # = AL_FALSE

#~ line: 263, skipped: 5 ~~~~~~

AL_INVALID_NAME = 0xA001

#~ line: 268, skipped: 5 ~~~~~~

AL_INVALID_ENUM = 0xA002

#~ line: 273, skipped: 5 ~~~~~~

AL_INVALID_VALUE = 0xA003

#~ line: 281, skipped: 8 ~~~~~~

AL_INVALID_OPERATION = 0xA004

#~ line: 287, skipped: 6 ~~~~~~

AL_OUT_OF_MEMORY = 0xA005

AL_VENDOR = 0xB001
AL_VERSION = 0xB002
AL_RENDERER = 0xB003
AL_EXTENSIONS = 0xB004

#~ line: 300, skipped: 7 ~~~~~~

AL_DOPPLER_FACTOR = 0xC000

#~ line: 305, skipped: 5 ~~~~~~

AL_DOPPLER_VELOCITY = 0xC001

#~ line: 310, skipped: 5 ~~~~~~

AL_DISTANCE_MODEL = 0xD000

#~ line: 314, skipped: 4 ~~~~~~

AL_INVERSE_DISTANCE = 0xD001
AL_INVERSE_DISTANCE_CLAMPED = 0xD002


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/altypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_openal import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alctypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ALCboolean(c_byte):
    """typedef ALCboolean"""

#~ line: 37, skipped: 6 ~~~~~~

class ALCubyte(c_char):
    """typedef ALCubyte"""
POINTER(ALCubyte).from_param = c_char_p.from_param

#~ line: 49, skipped: 12 ~~~~~~

class ALCint(c_int):
    """typedef ALCint"""

#~ line: 58, skipped: 9 ~~~~~~

class ALCsizei(c_uint):
    """typedef ALCsizei"""

ALCvoid = None # typedef ALCvoid

class ALCenum(c_int):
    """typedef ALCenum"""

ALC_INVALID = (-1)

ALC_FALSE = 0

ALC_TRUE = 1

ALC_NO_ERROR = 0 # = ALC_FALSE

ALC_MAJOR_VERSION = 0x1000
ALC_MINOR_VERSION = 0x1001
ALC_ATTRIBUTES_SIZE = 0x1002
ALC_ALL_ATTRIBUTES = 0x1003

ALC_DEFAULT_DEVICE_SPECIFIER = 0x1004
ALC_DEVICE_SPECIFIER = 0x1005
ALC_EXTENSIONS = 0x1006

ALC_FREQUENCY = 0x1007
ALC_REFRESH = 0x1008
ALC_SYNC = 0x1009

#~ line: 94, skipped: 5 ~~~~~~

ALC_INVALID_DEVICE = 0xA001

#~ line: 99, skipped: 5 ~~~~~~

ALC_INVALID_CONTEXT = 0xA002

#~ line: 107, skipped: 8 ~~~~~~

ALC_INVALID_ENUM = 0xA003

#~ line: 113, skipped: 6 ~~~~~~

ALC_INVALID_VALUE = 0xA004

#~ line: 119, skipped: 6 ~~~~~~

ALC_OUT_OF_MEMORY = 0xA005

#~ line: 129, skipped: 10 ~~~~~~

ALC_CONVERT_DATA_UPON_LOADING = 0xF001

#~ line: 134, skipped: 5 ~~~~~~

ALC_SPATIAL_RENDERING_QUALITY = 0xF002
ALC_SPATIAL_RENDERING_QUALITY_HIGH = 'rqhi'
ALC_SPATIAL_RENDERING_QUALITY_LOW = 'rdlo'

#~ line: 141, skipped: 5 ~~~~~~

ALC_MIXER_OUTPUT_RATE = 0xF003

#~ line: 149, skipped: 8 ~~~~~~

ALC_MIXER_MAXIMUM_BUSSES = 0xF004

#~ line: 155, skipped: 6 ~~~~~~

ALC_RENDER_CHANNEL_COUNT = 0xF005
ALC_RENDER_CHANNEL_COUNT_STEREO = 'rcst'
ALC_RENDER_CHANNEL_COUNT_MULTICHANNEL = 'rcmc'


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "/System/Library/Frameworks/OpenAL.framework/Versions/A/Headers/alctypes.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


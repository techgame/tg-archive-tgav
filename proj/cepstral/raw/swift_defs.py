#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_cepstral import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/swift_defs.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

swift_voice_desc_struct = c_void_p # Structure with empty _fields_
# typedef swift_voice
swift_voice = swift_voice_desc_struct

#~ line: 26, skipped: 6 ~~~~~~

swift_engine_struct = c_void_p # Structure with empty _fields_
# typedef swift_engine
swift_engine = swift_engine_struct

swift_port_struct = c_void_p # Structure with empty _fields_
# typedef swift_port
swift_port = swift_port_struct

cst_val_struct = c_void_p # Structure with empty _fields_
# typedef swift_params
swift_params = cst_val_struct

# typedef swift_val
swift_val = cst_val_struct

cst_wave_struct = c_void_p # Structure with empty _fields_
# typedef swift_waveform
swift_waveform = cst_wave_struct

# typedef swift_background_t
swift_background_t = c_void_p

swift_event_struct = c_void_p # Structure with empty _fields_
# typedef swift_event
swift_event = swift_event_struct

class swift_result_t(c_int):
    '''enum swift_result_t''' 
    SWIFT_SUCCESS = 0
    SWIFT_UNKNOWN_ERROR = -1
    SWIFT_UNIMPLEMENTED = -2
    SWIFT_INTERNAL_ERROR = -3
    SWIFT_INVALID_PARAM = -4
    SWIFT_INVALID_POINTER = -5
    SWIFT_OBJECT_NOT_FOUND = -6
    SWIFT_UNKNOWN_ENCODING = -7
    SWIFT_INTERRUPTED = -8
    SWIFT_INVALID_VOICE = -9
    SWIFT_FILE_ERROR = -10
    SWIFT_WRONG_EVENT = -11
    SWIFT_ENGINE_INUSE = -12
    SWIFT_NETWORK_ERROR = -13
    SWIFT_INVALID_KEY = -14
    SWIFT_QUEUE_FULL = -15
    SWIFT_TOKEN_TIMEOUT = -16
    lookup = {
        0: "SWIFT_SUCCESS",
        -1: "SWIFT_UNKNOWN_ERROR",
        -2: "SWIFT_UNIMPLEMENTED",
        -3: "SWIFT_INTERNAL_ERROR",
        -4: "SWIFT_INVALID_PARAM",
        -5: "SWIFT_INVALID_POINTER",
        -6: "SWIFT_OBJECT_NOT_FOUND",
        -7: "SWIFT_UNKNOWN_ENCODING",
        -8: "SWIFT_INTERRUPTED",
        -9: "SWIFT_INVALID_VOICE",
        -10: "SWIFT_FILE_ERROR",
        -11: "SWIFT_WRONG_EVENT",
        -12: "SWIFT_ENGINE_INUSE",
        -13: "SWIFT_NETWORK_ERROR",
        -14: "SWIFT_INVALID_KEY",
        -15: "SWIFT_QUEUE_FULL",
        -16: "SWIFT_TOKEN_TIMEOUT",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

#~ line: 65, skipped: 18 ~~~~~~

# typedef swift_result_t
swift_result_t = swift_result_t

class swift_status_t(c_int):
    '''enum swift_status_t''' 
    SWIFT_STATUS_UNKNOWN = -1
    SWIFT_STATUS_DONE = 0
    SWIFT_STATUS_RUNNING = 1
    SWIFT_STATUS_PAUSED = 2
    SWIFT_STATUS_QUEUED = 3
    lookup = {
        -1: "SWIFT_STATUS_UNKNOWN",
        0: "SWIFT_STATUS_DONE",
        1: "SWIFT_STATUS_RUNNING",
        2: "SWIFT_STATUS_PAUSED",
        3: "SWIFT_STATUS_QUEUED",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

#~ line: 74, skipped: 6 ~~~~~~

# typedef swift_status_t
swift_status_t = swift_status_t

class swift_event_t(c_int):
    '''enum swift_event_t''' 
    SWIFT_EVENT_NONE = 0
    SWIFT_EVENT_AUDIO = 1
    SWIFT_EVENT_ERROR = 2
    SWIFT_EVENT_SENTENCE = 4
    SWIFT_EVENT_PHRASE = 8
    SWIFT_EVENT_TOKEN = 16
    SWIFT_EVENT_WORD = 32
    SWIFT_EVENT_BOOKMARK = 64
    SWIFT_EVENT_SYLLABLE = 128
    SWIFT_EVENT_PHONEME = 256
    SWIFT_EVENT_START = 512
    SWIFT_EVENT_END = 1024
    SWIFT_EVENT_CANCELLED = 2048
    SWIFT_EVENT_ALL = -1
    SWIFT_EVENT_NOW = -1
    lookup = {
        0: "SWIFT_EVENT_NONE",
        1: "SWIFT_EVENT_AUDIO",
        2: "SWIFT_EVENT_ERROR",
        4: "SWIFT_EVENT_SENTENCE",
        8: "SWIFT_EVENT_PHRASE",
        16: "SWIFT_EVENT_TOKEN",
        32: "SWIFT_EVENT_WORD",
        64: "SWIFT_EVENT_BOOKMARK",
        128: "SWIFT_EVENT_SYLLABLE",
        256: "SWIFT_EVENT_PHONEME",
        512: "SWIFT_EVENT_START",
        1024: "SWIFT_EVENT_END",
        2048: "SWIFT_EVENT_CANCELLED",
        -1: "SWIFT_EVENT_ALL",
        -1: "SWIFT_EVENT_NOW",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

#~ line: 93, skipped: 16 ~~~~~~

# typedef swift_event_t
swift_event_t = swift_event_t

class swift_voice_retention_policy_t(c_int):
    '''enum swift_voice_retention_policy_t''' 
    SWIFT_VOICE_RETAIN_FOREVER = 0
    SWIFT_VOICE_RETAIN_PORT_USAGE = 1
    SWIFT_VOICE_RETAIN_NONE = 2
    lookup = {
        0: "SWIFT_VOICE_RETAIN_FOREVER",
        1: "SWIFT_VOICE_RETAIN_PORT_USAGE",
        2: "SWIFT_VOICE_RETAIN_NONE",
        }
    rlookup = dict([(v,k) for k,v in lookup.items()])
    def __repr__(self): return str(self)
    def __str__(self): 
        return self.lookup.get(self.value) or str(self.value)

#~ line: 100, skipped: 4 ~~~~~~

# typedef swift_voice_retention_policy_t
swift_voice_retention_policy_t = swift_voice_retention_policy_t


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/swift_defs.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


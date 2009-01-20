/* -*- c-file-style: "linux" -*- */
/* swift_defs.h: Forward declarations and enumerations for Swift TTS.
 *
 * Copyright (c) 2004-2006 Cepstral LLC.  All rights reserved.
 *
 * Redistribution of this file, in whole or in part, with or without
 * modification, is not allowed without express written permission of
 * the copyright holder.
 */

#ifndef _SWIFT_DEFS_H_
#define _SWIFT_DEFS_H_

/**
 * \file swift_defs.h
 * Types and enumerations for the Swift text-to-speech API.
 **/

/** Description of a voice available to the system. */
typedef struct swift_voice_desc_struct swift_voice;

/** A linked-list of swift_voice structs **/
typedef struct swift_voice_list_desc_struct swift_voice_list;

/** TTS Engine Object through which ports may be open. */
typedef struct swift_engine_struct swift_engine;

/** Swift Port Object through which speech synthesis is performed. */
typedef struct swift_port_struct swift_port;

/** Parameter list (key-value pairs). */
typedef struct cst_val_struct swift_params;

/** Generic value object. */
typedef struct cst_val_struct swift_val;

/** Waveform object. */
typedef struct cst_wave_struct swift_waveform;

/** ID of background synthesis job. */
typedef void * swift_background_t;

/** Synthesis event. */
typedef struct swift_event_struct swift_event;

/** Result (error/success) codes for API functions. */
typedef enum swift_result_t {
    SWIFT_SUCCESS           = 0,   /**< Operation succeeded. */
    SWIFT_UNKNOWN_ERROR     = -1,  /**< Unknown error. */
    SWIFT_UNIMPLEMENTED     = -2,  /**< Function is not (yet) implemented. */
    SWIFT_INTERNAL_ERROR    = -3,  /**< Engine internal error. */
    SWIFT_INVALID_PARAM     = -4,  /**< Invalid parameter. */
    SWIFT_INVALID_POINTER   = -5,  /**< Invalid NULL pointer in parameter. */
    SWIFT_OBJECT_NOT_FOUND  = -6,  /**< File, voice, port, etc... not found. */
    SWIFT_UNKNOWN_ENCODING  = -7,  /**< Unknown text/audio encoding. */
    SWIFT_INTERRUPTED       = -8,  /**< Operation interrupted. */
    SWIFT_INVALID_VOICE     = -9,  /**< Voice not present or corrupted. */
    SWIFT_FILE_ERROR        = -10, /**< A file could not be read/written. */
    SWIFT_WRONG_EVENT       = -11, /**< Wrong event type for the data requested.*/
    SWIFT_ENGINE_INUSE      = -12, /**< Engine still in use by some port. */
    SWIFT_NETWORK_ERROR     = -13, /**< Network communication failed. */
    SWIFT_INVALID_KEY       = -14, /**< An invalid license key was specified. */
    SWIFT_QUEUE_FULL        = -15, /**< The utterance queue is full. */
    SWIFT_TOKEN_TIMEOUT     = -16  /**< Get concurrecny token timed out. */
} swift_result_t;

/** Status of a background job. */
typedef enum swift_status_t {
    SWIFT_STATUS_UNKNOWN = -1, /**< No such job. */
    SWIFT_STATUS_DONE    = 0,  /**< Job has completed. */
    SWIFT_STATUS_RUNNING = 1,  /**< Job is currently running. */
    SWIFT_STATUS_PAUSED  = 2,  /**< Job is currently paused. */
    SWIFT_STATUS_QUEUED  = 3   /**< Job is waiting to run. */
} swift_status_t;

/** Type of event for a callback (also used as a mask to request events). */
typedef enum swift_event_t {
    SWIFT_EVENT_NONE        = 0,
    SWIFT_EVENT_AUDIO       = (1<<0),
    SWIFT_EVENT_ERROR       = (1<<1),
    SWIFT_EVENT_SENTENCE    = (1<<2),
    SWIFT_EVENT_PHRASE      = (1<<3),
    SWIFT_EVENT_TOKEN       = (1<<4),
    SWIFT_EVENT_WORD        = (1<<5),
    SWIFT_EVENT_BOOKMARK    = (1<<6),
    SWIFT_EVENT_SYLLABLE    = (1<<7),
    SWIFT_EVENT_PHONEME     = (1<<8),
    SWIFT_EVENT_START       = (1<<9),     /* Synthesis job started. */
    SWIFT_EVENT_END         = (1<<10),    /* Synthesis job completed. */
    SWIFT_EVENT_CANCELLED   = (1<<11),    /* User cancel request completed. */
    SWIFT_EVENT_ALL         = 0xffffffff,
    SWIFT_EVENT_NOW         = 0xffffffff  /* Used for halt/pause. */
} swift_event_t;

/** Policy for retention of voice settings and parameters. */
typedef enum swift_voice_retention_policy_t {
    SWIFT_VOICE_RETAIN_FOREVER       = 0,
    SWIFT_VOICE_RETAIN_PORT_USAGE    = (1<<0),
    SWIFT_VOICE_RETAIN_NONE          = (1<<1)
} swift_voice_retention_policy_t;

/** No background job. */
#define SWIFT_ASYNC_NONE ((swift_background_t)0)
/** Any background job. */
#define SWIFT_ASYNC_ANY ((swift_background_t)-1)
/** Currently-running background job. */
#define SWIFT_ASYNC_CURRENT ((swift_background_t)-3)

/** Determine if an operation (in parentheses) failed. */
#define SWIFT_FAILED(r) ((r) < 0)

/* String encoding types. */
/** UTF-8 Encoding type **/
#define SWIFT_UTF8 "utf-8"
/** ASCII Encoding type **/
#define SWIFT_ASCII "us-ascii"
/** UTF-16 Encoding type **/
#define SWIFT_UTF16 "utf-16"
/** Unicode Encoding type (maps to UTF-16) **/
#define SWIFT_UNICODE "utf-16"
/** iso8859-1 Latin-1 Encoding type **/
#define SWIFT_LATIN_1 "iso8859-1"
/** iso8859-1 Encoding type **/
#define SWIFT_ISO_8859_1 "iso8859-1"
/** iso8859-15 Latin-9 Encoding type **/
#define SWIFT_LATIN_9 "iso8859-15"
/** iso8859-15 Encoding type **/
#define SWIFT_ISO_8859_15 "iso8859-15"
/** Default Encoding type - Latin 1 - iso8859-1 **/
#define SWIFT_DEFAULT_ENCODING SWIFT_LATIN_1

#endif /* _SWIFT_DEFS_H_ */

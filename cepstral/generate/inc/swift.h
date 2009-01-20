/* swift.h: Swift API for text-to-speech
 *
 * Copyright (c) 2002-2006 Cepstral LLC.  All rights reserved.
 *
 * Redistribution of this file, in whole or in part, with or without
 * modification, is not allowed without express written permission of
 * the copyright holder.
 */

#ifndef _SWIFT_H_
#define _SWIFT_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * \file swift.h
 * Main include file for Swift text-to-speech engine.
 **/

#include "swift_exports.h"
#include "swift_defs.h"
#include "swift_params.h"

/* swift_version strings exposed through SWIFT_API */
/** Contains the name of the Swift engine, as a string. */
SWIFT_API extern const char *swift_engine_name;
/** Contains the version of the Swift library, as a string. */
SWIFT_API extern const char *swift_version;
/** Contains the date of the Swift library, as a string. */
SWIFT_API extern const char *swift_date;
/** Contains the platform for which the Swift library was built,
 *  as a string. */
SWIFT_API extern const char *swift_platform;

/**
 * Open an instance of the Swift TTS engine.
 *
 * @param params A swift_params object with engine initialization parameters.
 *  <br>Parameters recognized are:
 *  - config/voice-path (string): A colon (most platforms) or semicolon
 *    (Windows) delimited list of directories to be searched for voices.
 *  - config/default-voice (string): The name of the default voice to
 *    use for text-to-speech.
 *  - NOTE: The engine retains ownership of this object, so you should not
 *          call swift_params_delete() on it.
 * @return a new engine if successful, NULL for failure.
 **/
SWIFT_API
swift_engine * SWIFT_CALLCONV swift_engine_open(swift_params *params);

/**
 * Close and delete a Swift TTS engine.
 *
 * @param engine A swift_engine object
 * @return       SWIFT_SUCCESS if successful, SWIFT_ENGINE_INUSE if engine
 *               still has open ports associated with it
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_engine_close(swift_engine *engine);

/**
 * Set the voice retention policy for the engine.  Voice settings and
 * parameters will be allocated as the voice is loaded.  The retention
 * policy dictates when this information will be released, allowing for
 * the adjustment of voice-load overhead.
 *
 * NOTE: This function currently has no effect.  The engine uses a retention
 *       policy of SWIFT_VOICE_RETAIN_FOREVER regardless of the actual setting.
 *
 * @param engine A swift_engine object
 * @param policy A voice retention (i.e.: garbage collection policy).  May
 * be one of:
 *  - SWIFT_VOICE_RETAIN_FOREVER:  retain voice information for the
 *    lifetime of the engine.
 *  - SWIFT_VOICE_RETAIN_PORT_USAGE:  retain voice information for the
 *    lifetime of the ports using it.
 *  - SWIFT_VOICE_RETAIN_NONE:  retain voice information only as long as
 *    the voice is in use.
 * @return void
 **/
SWIFT_API
void SWIFT_CALLCONV
swift_engine_set_voice_retention_policy(swift_engine *engine,
                                        swift_voice_retention_policy_t policy);

/**
 * Get the voice retention policy for the engine.
 *
 * @param engine A swift_engine object
 * @return The voice retention (i.e.: garbage collection policy).  May be
 * one of:
 *  - SWIFT_VOICE_RETAIN_FOREVER:  retain voice information for the
 *    lifetime of the engine.
 *  - SWIFT_VOICE_RETAIN_PORT_USAGE:  retain voice information for the
 *    lifetime of the ports using it.
 *  - SWIFT_VOICE_RETAIN_NONE:  retain voice information only as long as
 *    the voice is in use.
 **/
SWIFT_API
swift_voice_retention_policy_t SWIFT_CALLCONV
swift_engine_get_voice_retention_policy(swift_engine *engine);

/**
 * Open a new TTS port.
 *
 * @param engine An instance of a swift engine from swift_engine_open
 * @param params A swift_params object with port initialization
 *               parameters and port-wide synthesis parameters.
 * - NOTE: The port retains ownership of this object, so you should not
 *         call swift_params_delete() on it.
 * @return The new port, or NULL for failure.
 **/
SWIFT_API
swift_port * SWIFT_CALLCONV swift_port_open(swift_engine *engine,
                                            swift_params *params);

/**
 * Close a TTS port and dispose of its resources.
 *
 * @param port The swift_port to close.
 * @return     SWIFT_SUCCESS on successful close, or one of several
 *             swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_close(swift_port *port);

/**
 * User-Defined callback function type (for audio, events, errors, etc).
 * Pointer to this function is passed to swift_port_set_callback().
 *
 * @param event The event object this callback is being called for.
 * @param type  The type of this event.
 * @param udata The user data pointer passed to swift_port_set_callback().
 * @return      Your callback should return SWIFT_SUCCESS (or 0), unless
 *              you wish to halt synthesis immediately, in which case
 *              you should return SWIFT_INTERRUPTED.
 **/
typedef swift_result_t (*swift_callback_t)(swift_event *event,
                                           swift_event_t type,
                                           void *udata);

/**
 * Set a callback for a port.  This allows a user-defined function to
 * handle swift_events as they are sent by the engine.
 *
 * @param port     The port for which this callback is to be set.
 * @param callback The callback function to set up, or NULL to remove
 *                 an existing callback function.
 * @param mask     A bitmask of events you wish this function to be called
 *                 for (see enum swift_event_t).
 * @param udata    User Data - pointer to anything you want to pass to the
 *                 callback function.
 * @return         The previous callback function, or NULL if none.
 **/
SWIFT_API
swift_callback_t SWIFT_CALLCONV
swift_port_set_callback(swift_port *port, swift_callback_t callback,
                        unsigned int mask, void *udata);

/**
 * Get the string name of the given event type.
 *
 * @param type The event type of which to get the string name.
 * @return     The string name of the event, or NULL if no such event.
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_event_type_get_name(swift_event_t type);

/**
 * Get the event type for the given string name.
 *
 * @param name The string event name for which to get the type.
 * @return     The swift_event_t corresponding to the name, or 0 if none.
 **/
SWIFT_API
swift_event_t SWIFT_CALLCONV swift_event_name_get_type(const char *name);

/**
 * Get the swift_port and swift_background_t thread ID from which an event
 * was generated.
 *
 * @param event The event for which to get the information.
 * @param port  Output - the swift_port from which the event was generated.
 * @param async Output - the swift_background_t thread ID from wich the event
 *              was generated.
 * @return      SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if event is NULL
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_event_get_context(swift_event *event, swift_port **port,
                        swift_background_t *async);

/**
 * Get the timepoints for an event of type sentence, phrase, word, token,
 * syllable, or phoneme.
 *
 * @param event The event for which to get the information.
 * @param start Output - The starting time of the event with relation to
 *              the start of synthesis.
 * @param len   Output - The length of the event
 * @return      SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if event is NULL
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_times(swift_event *event,
                                                    float *start, float *len);

/**
 * Get the text positions for an event of type sentence, phrase, word, or
 * token.
 *
 * @param event The event for which to get the information.
 * @param start Output - The starting position in the synthesis input text
 *              corresponding to the event.
 * @param len   Output - The length in characters of the text corresponding
 *              to the event.
 * @return      SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if event is NULL,
 *              or SWIFT_WRONG_EVENT if event type isn't
 *              SWIFT_EVENT_[SENTENCE/PHRASE/TOKEN/WORDBOOKMARK].
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_textpos(swift_event *event,
                                                      int *start, int *len);

/**
 * Get the text string associated with an event of type sentence, phrase,
 * word, or phoneme.
 *
 * @param event The event for which to get the information.
 * @param text  Output - pointer to the text associated with the event.
 * @return      SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_text(swift_event *event,
                                                   char **text);

/**
 * Get the audio buffer for an audio event.
 *
 * @param event  The event for which to get the audio buffer.
 * @param buf    Output - The buffer that will receive the waveform.
 * @param nbytes Output - The number of bytes copied into the buffer.
 * @return       SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_audio(swift_event *event,
                                                    void **buf, int *nbytes);

/**
 * Get the swift_waveform structure for an audio event.
 *
 * @param event       The event for which to get the audio buffer.
 * @param wave        Input/Output - A pointer to a wave structure.
 * @param concatenate Boolean - If true, audio will be concatenated onto the
 *                    wave structure passed in.  If false, the synthesized
 *                    audio will overwrite any audio that may already be in
 *                    the wave structure.
 * @return            SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_wave(swift_event *event,
                                                   swift_waveform **wave,
                                                   int concatenate);

/**
 * Get the error code and string message for an error event.
 *
 * @param event  The error event for which to get the error information.
 * @param rv     Output - The error code of the error event.
 * @param errmsg Output - The string message for the error event.
 * @return       SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if event is
 *               NULL, or SWIFT_WRONG_EVENT if event type is not
 *               SWIFT_EVENT_ERROR.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_event_get_error(swift_event *event,
                                                    swift_result_t *rv,
                                                    const char **errmsg);

/**
 * Set a port parameter.
 *
 * @param port  The port being modified.
 * @param name  The name of the parameter.
 * @param val   The value (as a swift_val) of the parameter.
 * @param async The background job in which to set this parameter.
 * @return      SWIFT_SUCCESS if successful, SWIFT_INVALID_PARAM if
 *              no such parameter exists.
 *
 * This function sets a synthesis parameter in a port, and optionally
 * sets it in some or all of that port's running background jobs.  The
 * parameter and value described by NAME and VAL must be valid,
 * i.e. NAME must specify a parameter that exists, and VAL must be a
 * proper value for that parameter.  swift_param_validate() is used to
 * check this, against swift_param_descriptors.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_set_param(swift_port *port,
                                                   const char *name,
                                                   swift_val *val,
                                                   swift_background_t async);
/**
 * Set a list of parameters in a port.
 *
 * @param port   The port on which to set the parameters list.
 * @param params The list of parameters to set on the port.
 * @param async  The background job, if any, to set this paramters
 *               list in.
 * @return       SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t  SWIFT_CALLCONV swift_port_set_params(swift_port *port,
                                                     swift_params *params,
                                                     swift_background_t async);

/** Set an integer or boolean parameter in a port. */
#define swift_port_set_param_int(p,n,i,a) \
    swift_port_set_param(p,n,swift_val_int((int)(i)),a)

/** Set a floating-point parameter in a port. */
#define swift_port_set_param_float(p,n,f,a) \
    swift_port_set_param(p,n,swift_val_float((float)(f)),a)

/** Set a string parameter in a port. */
#define swift_port_set_param_string(p,n,s,a) \
    swift_port_set_param(p,n,swift_val_string((const char *)(s)),a)

/**
 * Get a pointer to the first (default) voice matching params.
 *
 * @param port            The port on which to search for voices.
 * @param search_criteria A semicolon-separated list of attribute=value
 *                        pairs required from the voice, or NULL to return
 *                        the default voice.
 * @param order_criteria  A semicolon-separated list of attribute=value
 *                        pairs used to set a loose sorting order on the
 *                        list of those that match match the seach_criteria.
 *                        Voices matching the order_criteria are placed at
 *                        the top of the list.
 * @return                A descriptor for the voice, or NULL if not found.
 *
 * The search_criteria and order_criteria arguments are strings containing
 * one or more expressions which are to be tested against the voice's
 * attributes.  These expressions are of the form ATTR=VALUE,
 * ATTR<VALUE, or ATTR>VALUE.  For example, to search for female
 * voices only, you would call swift_port_find_first_voice(port,
 * "speaker/gender=female", NULL).
 *
 * The order_criteria argument controls the ordering of the voices returned.
 * 
 * Currently available attributes to use for search_criteria and order_criteria
 * are:
 *  - name
 *  - path
 *  - version
 *  - sample-rate
 *  - license/key
 *  - language/tag
 *  - language/name
 *  - language/version
 *  - lexicon/name
 *  - lexicon/version
 *  - speaker/name
 *  - speaker/gender
 *  - speaker/age
 *
 * Negation of an expression is allowed, simply prepend a '!'.
 * Disjunctions and conjunctions are also allowed, using '|' and '&'
 * respectively.  So to find voices for either US or UK English, you
 * could call swift_port_find_first_voice(port,
 * "language/tag=en-US|language/tag=en-UK");
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV
swift_port_find_first_voice(swift_port *port, const char *search_criteria,
                            const char *order_criteria);

/**
 * Get the next voice in the list of voices matching the attributes
 * passed to swift_find_first_voice().
 *
 * @param port The port on which to find the next voice.
 * @return     The next matching voice, or NULL if the end of the list is
 *             reached.
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV swift_port_find_next_voice(swift_port *port);

/** Rewind the voice pointer to the first voice returned by
 *  swift_port_find_find_first_voice().
 *
 * @param port The port on which to rewind the voice list.
 * @return     The first voice in the list returned by
 *             swift_port_find_first_voice().
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV swift_port_rewind_voices(swift_port *port);

/**
 * Load a voice and set it as the port's current voice.
 *
 * @param port  The port on which to set the voice.
 * @param voice The voice to load and set as the port's current voice.
 * @return      SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_set_voice(swift_port *port,
                                                   swift_voice *voice);

/**
 * Load the voice referenced by name and set it as the port's current
 * voice.
 *
 * @param port The port on which to set the voice.
 * @param name The name of the voice to load and set on the port.
 * @return     The voice if it was successfully loaded, else NULL.
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV swift_port_set_voice_by_name(swift_port *port,
                                                          const char *name);

/**
 * Load a voice from a directory and set it as the port's current voice.
 *
 * @param port The port on which to set the voice.
 * @param dir  The directory from which to load the voice.
 * @return     The voice if it was successfully loaded, else NULL.
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV swift_port_set_voice_from_dir(swift_port *port,
                                                           const char *dir);

/**
 * Get a pointer to the current voice.
 *
 * @param port The port from which to get the current voice.
 * @return     The current voice from the port.  This could be NULL
 *             if no voice was ever set for the port.
 **/
SWIFT_API
swift_voice * SWIFT_CALLCONV swift_port_get_current_voice(swift_port *port);

/**
 * Get an attribute from a voice.
 *
 * @param voice The voice from which to get the attribute.
 * @param attr  The name of the attribute to get from the voice.
 * @return      The value of the requested param, or NULL.
 *
 * Currently available attributes are:
 *  - id
 *  - name
 *  - path
 *  - version
 *  - buildstamp
 *  - sample-rate
 *  - license/key
 *  - language/tag
 *  - language/name
 *  - language/version
 *  - lexicon/name
 *  - lexicon/version
 *  - speaker/name
 *  - speaker/gender
 *  - speaker/age
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_voice_get_attribute(swift_voice *voice,
                                                      const char *attr);

/**
 * Get all attributes from a voice.
 * @param voice      The voice to get attributes for.
 * @param out_params Output - A swift_params in which to store the
 *                   attributes.
 * @return           SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if voice
 *                   or out_params is NULL
 *
 * After getting the attributes, you can use swift_params_dump() to
 * print them out or swift_params_foreach() to iterate over them.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_voice_get_attributes(swift_voice *voice, swift_params *out_params);

/**
 * Load custom lexicon entries into a voice from a file.
 *
 * @param voice The voice for which to load the lexicon.
 * @param file  The file from which to load the lexicon.
 * @return      SWIFT_SUCCESS, or SWIFT_FILE_ERROR if file could not be opened.
 *
 * When synthesizing text, the search order for lexicon entries is:
 *   <br>1. Lexicons loaded with this API call, most recent first.
 *   <br>2. A lexicon.txt file in the voice directory
 *   <br>3. The built-in lexicon
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_voice_load_lexicon(swift_voice *voice,
                                                       const char *file);

/**
 * Returns the string encoding of the current active voice on the port.
 *
 * @param port The port from which to get the language encoding.
 * @return     The port's active voice's text encoding, or NULL if that
 *             can't be attained.
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_port_language_encoding(swift_port *port);

/**
 * Load a special effects output chain into a port from a file.
 *
 * @param port The port onto which to load the sfx chain.
 * @param file The file to from which to load, or NULL to remove the
 *             current sfx chain without loading a new one.
 * @return     SWIFT_SUCCESS, or SWIFT_FILE_ERROR if file could not be opened.
 *
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_load_sfx(swift_port *port,
                                                  const char *file);

/**
 * Query the status of a background job.
 *
 * @param port  The port to query for background job status.
 * @param async The background job to look for.  Can one of:
 *   - A swift_background_t thread ID
 *   - SWIFT_ASYNC_ANY to get the status of any job
 *   - SWIFT_ASYNC_CURRENT to get the status of the currently running job
 * @return One of:
 *   - SWIFT_STATUS_UNKNOWN if the job specified doesn't exist
 *   - SWIFT_STATUS_DONE if it has completed
 *   - SWIFT_STATUS_RUNNING if it is currently running
 *   - SWIFT_STATUS_QUEUED if it is scheduled to run in the future.
 **/
SWIFT_API
swift_status_t SWIFT_CALLCONV swift_port_status(swift_port *port,
                                                swift_background_t async);

/**
 * Wait for a background job (and any jobs ahead of it) to complete.
 *
 * @param port  The port on which to wait for background jobs.
 * @param async The background job to wait for.  Can be one of:
 *   - A swift_background_t thread ID
 *   - SWIFT_ASYNC_ANY to wait for the completion of any job
 *   - SWIFT_ASYNC_CURRENT to wait for the completion of the currently
 *     running job
 * @return      SWIFT_SUCCESS
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_wait(swift_port *port,
                                              swift_background_t async);

/**
 * Stop and cancel a background job, waiting for any jobs ahead of it
 * on the port to complete, and cancelling any jobs after it.
 *
 * @param port  The port to on which to stop background jobs.
 * @param async The background job to stop.  Can be one of:
 *   - A swift_background_t thread ID
 *   - SWIFT_ASYNC_ANY to stop any job
 *   - SWIFT_ASYNC_CURRENT to stop the currently running job
 * @param place The event (e.g. word, sentence, syllable) to wait for
 *              before halting.  Use SWIFT_EVENT_NOW for immediate halt.
 *   - NOTE: This parameter is currently ignored.  Halting always occurs
 *           immediately.
 *   - NOTE: No swift_events will be sent to your swift_callback_t function
 *           for the given background job after swift_port_stop is called.
 * @return      SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if port or its
 *              thread handler is NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_stop(swift_port *port,
                                              swift_background_t async,
                                              swift_event_t place);

/**
 * Pause a background job.  If the job you want to pause hasn't started yet
 * and is in the queue, then any jobs ahead of it complete first.
 *
 * @param port  The port on which to pause a background job.
 * @param async The background job to pause.  Can be one of:
 *   - A swift_background_t thread ID
 *   - SWIFT_ASYNC_ANY for any job
 *   - SWIFT_ASYNC_CURRENT for the currently running job
 * @param place The event (e.g. word, sentence, syllable) to wait for
 *              before halting.  Use SWIFT_EVENT_NOW for immediate pause.
 *   - NOTE: This parameter is currently ignored.  Halting always occurs
 *           immediately.
 * @return      SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_pause(swift_port *port,
                                               swift_background_t async,
                                               swift_event_t place);

/**
 * Synthesize text on the given port.
 *
 * @param port     The port on which to perform synthesis.
 * @param text     Text to synthesize.
 * @param nbytes   Length, in bytes, of text.  If 0, this paramter is ignored.
 * @param encoding Character encoding of the input text.  If NULL, the default
 *                 encoding is assumed.
 * @param async    Output - Thread ID of the background job on which to run
 *                 this synthesis call.<br>If NULL, synthesis occurs
 *                 synchronously, thus blocking until complete.
 * @param params   The parameters to be used for this synthesis call. Can
 *                 be NULL.
 * - NOTE: The utterance retains ownership of this object, so you should not call
 *         swift_params_delete() on it.
 * @return         SWIFT_SUCCESS, or one of several swift_result_t values.
 *
 * - NOTE: If a voice was not explicitely set on the port via
 *         swift_port_set_voice(), swift_port_set_voice_by_name(), or
 *         swift_port_set_voice_from_dir(), then this call implicitely loads
 *         the default voice and sets it as the port's current voice.
 *
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_speak_text(swift_port *port,
                                                    const void *text,
                                                    int nbytes,
                                                    const char *encoding,
                                                    swift_background_t *async,
                                                    swift_params *params);

/**
 * Synthesize text from a file on the given port.
 *
 * @param port     The port on which to perform synthesis.
 * @param path     The path to the file from which to synthesize the text.
 * @param encoding Character encoding of the input text.  If NULL, the default
 *                 encoding is assumed.
 * @param async    Output - Thread ID of the background job on which to run
 *                 this synthesis call.<br>If NULL, synthesis occurs
 *                 synchronously, thus blocking until complete.
 * @param params   The parameters to be used for this synthesis call. Can
 *                 be NULL.
 * - NOTE: The utterance retains ownership of this object, so you should not call
 *         swift_params_delete() on it.
 * @return         SWIFT_SUCCESS, or one of several swift_result_t values.
 *
 * - NOTE: If a voice was not explicitely set on the port via
 *         swift_port_set_voice(), swift_port_set_voice_by_name(), or
 *         swift_port_set_voice_from_dir(), then this call implicitely loads
 *         the default voice and sets it as the port's current voice.
 **/
/* FIXME: Windows has wide character filenames */
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_speak_file(swift_port *port,
                                                    const char *path,
                                                    const char *encoding,
                                                    swift_background_t *async,
                                                    swift_params *params);

/**
 * Get Performance Statistics for the most recently completed synthesis call
 * on the given port.
 *
 * This information is written to the swift_port at the end of each synthesis
 * call.  As such, the information is volatile; It only lasts until the next
 * synthesis call is completed.  If you are using this function, make sure that
 * you call it after a synthesis call has completed, but before the next
 * synthesis call has completed to ensure you are getting reliable data.
 *
 * @param port             The port for which to get performance statistics.
 * @param uttcount         Output - Number of utterances synthesized during the
 *                         most recently completed synthesis call.
 * @param avg_startup_time Output - Average time to begin speaking each
 *                         utterance during the most recently completed
 *                         synthesis call.
 * @param audiotime        Output - Total length in seconds of audio generated
 *                         during the most recently completed synthesis call.
 * @param realtime         Output - Total length in seconds required to
 *                         complete the most recently completed synthesis call.
 * @param speed_ratio      Output - Ratio of audio time to real time.  Shows 
 *                         how much faster synthesis was completed compared to
 *                         the amount of audio generated.
 * @return                 SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if port
 *                         is NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_get_perfstats(swift_port *port,
                                                       int *uttcount,
                                                       float *avg_startup_time,
                                                       float *audiotime,
                                                       float *realtime,
                                                       float *speed_ratio);

/**
 * Synthesize text string or text from a file, and return the corresponding
 * phonemes and event time information.
 *
 * @param port     The port on which to perform the synthesis.
 * @param text     The text to synthesize, or the path to the file from which
 *                 to synthesize the text.
 * @param nbytes   Length, in bytes, of text.  If 0, this paramter is ignored.
 * @param encoding Character encoding of the input text.  If NULL, the default
 *                 encoding is assumed.
 * @param is_file  Boolean - Set this to true if text is the path to a file.
 * @param params   The parameters to be used for this synthesis call. Can
 *                 be NULL.
 * - NOTE: The utterance retains ownership of this object, so you should not call
 *         swift_params_delete() on it.
 * @return         A string containing a list of the phonemes and event time
 *                 information for the synthesized speech.
 *
 * - NOTE: If a voice was not explicitely set on the port via
 *         swift_port_set_voice(), swift_port_set_voice_by_name(), or
 *         swift_port_set_voice_from_dir(), then this call implicitely loads
 *         the default voice and sets it as the port's current voice.
 *
 * - NOTE: This function DOES NOT WORK as of Swift v3.1.0.
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_port_get_phones(swift_port *port,
                                                  const void *text,
                                                  int nbytes,
                                                  const char *encoding,
                                                  int is_file,
                                                  swift_params *params);

/**
 * Synthesize text string or text from file to a swift_waveform.
 *
 * @param port     The port on which to perform the synthesis.
 * @param text     The text to synthesize, or the path to the file from which
 *                 to synthesize the text.
 * @param nbytes   Length, in bytes, of text.  If 0, this paramter is ignored.
 * @param encoding Character encoding of the input text.  If NULL, the default
 *                 encoding is assumed.
 * @param is_file  Boolean - Set this to true if text is the path to a file.
 * @param params   The parameters to be used for this synthesis call. Can
 *                 be NULL.
 * - NOTE: The utterance retains ownership of this object, so you should not call
 *         swift_params_delete() on it.
 * @return         Pointer to the swift_waveform containing the synthesized
 *                 speech.
 *
 * - NOTE: If a voice was not explicitely set on the port via
 *         swift_port_set_voice(), swift_port_set_voice_by_name(), or
 *         swift_port_set_voice_from_dir(), then this call implicitely loads
 *         the default voice and sets it as the port's current voice.
 **/
SWIFT_API
swift_waveform * SWIFT_CALLCONV swift_port_get_wave(swift_port *port,
                                                    const void *text,
                                                    int nbytes,
                                                    const char *encoding,
                                                    int is_file,
                                                    swift_params *params);

/**
 * Play a swift_waveform through a port.
 *
 * @param port   The port on which to play the waveform.
 * @param wav    The waveform to play on the port.
 * @param async  Output - Thread ID of the background job on which to run this
 *               synthesis call.<br>If NULL, synthesis occurs synchronously,
 *               thus blocking until complete.
 * @param params The parameters to be used for this synthesis call. Can
 *               be NULL.
 * - NOTE: The utterance retains ownership of this object, so you should not call
 *         swift_params_delete() on it.
 * @return       SWIFT_SUCCESS, or one of several swift_result_t values.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_port_play_wave(swift_port *port,
                                                   const swift_waveform *wav,
                                                   swift_background_t *async,
                                                   swift_params *params);

/**
 * Create a new, empty waveform object.
 *
 * @return A new, empty waveform object.
 **/
SWIFT_API
swift_waveform * SWIFT_CALLCONV swift_waveform_new(void);

/**
 * Load a swift_waveform object from an audio file.
 *
 * @param filename     The path to the audio file to open.
 * @param format       The format of the audio file (see note below).
 * @param encoding     The encoding of the file.  One of "pcm16", "pcm8",
 *                     "ulaw", "alaw".  This is only used if the format is
 *                     "raw", "le", or "be", and can be NULL otherwise.
 * @param sample_rate  The sampling rate of the audio data.  This is only used
 *                     if the format is "raw", "le", or "be", and will be
 *                     ignored otherwise.
 * @param num_channels The number of interleaved channels in the audio data.
 *                     This is only used if the formate is "raw", "le", or
 *                     "be", and will be ignored otherwise.
 * @return             The swift_wavform object that has been populated with
 *                     data from the audio file.
 *
 * NOTE: The format argument is one of:
 *  - riff: Microsoft RIFF (WAV) file
 *  - snd:  Sun/NeXT .au (SND) format.
 *  - raw:  unheadered audio data, native byte order
 *  - le:   unheadered audio data, little-endian
 *  - be:   unheadered audio data, big-endian
 *  - NULL: the filename's extension will determine the format.
 **/
SWIFT_API
swift_waveform * SWIFT_CALLCONV swift_waveform_open(const char *filename,
                                                    const char *format,
                                                    const char *encoding,
                                                    int sample_rate,
                                                    int num_channels);

/**
 * Print detailed information about a swift_waveform.
 *
 * @param wave The swift_waveform about which to print detailed information.
 * @return     Void
 **/
SWIFT_API
void SWIFT_CALLCONV swift_waveform_print(swift_waveform *wave);

/**
 * Close a swift_waveform and deallocate associated memory and resources.
 *
 * @param wave The swift_waveform to close.
 * @return     SWIFT_SUCCESS
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_close(swift_waveform *wave);

/**
 * Save a swift_waveform to an audio file.
 *
 * @param wave     The swift_waveform to save to the file.
 * @param filename The path to the file in which to save the swift_waveform data.
 * @param format   The format of the swift_waveform (see note below).
 * @return         SWIFT_SUCCESS, or one of several swift_result_t values.
 *
 * NOTE: The format argument is one of:
 *  - riff: Microsoft RIFF (WAV) file
 *  - snd:  Sun/NeXT .au (SND) format.
 *  - raw:  unheadered audio data, native byte order
 *  - le:   unheadered audio data, little-endian (LSB first)
 *  - be:   unheadered audio data, big-endian (MSB first)
 *  - NOTE: that not all encoding types may be supported by all formats.
 *          For instance, SND doesn't support A-Law.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_save(const swift_waveform *wave,
                                                  const char *filename,
                                                  const char *format);

/**
 * Get the sample rate of a swift_waveform.
 *
 * @param wave The swift_waveform from which to get the sample rate.
 * @return     The sample rate of the waveform.
 **/
SWIFT_API
int SWIFT_CALLCONV swift_waveform_get_sps(const swift_waveform *wave);

/**
 * Get the encoding type for a swift_waveform.
 *
 * @param wave The swift_waveform from which to get the encoding type.
 * @return     The encoding type, one of:
 *   - "pcm16" - 16-bit signed linear PCM
 *   - "pcm8"  - 8-bit unsigned linear PCM
 *   - "ulaw"  - µ-Law (8-bit by definition)
 *   - "alaw"  - A-Law (8-bit by definition)
 **/
SWIFT_API
const char * SWIFT_CALLCONV
swift_waveform_get_encoding(const swift_waveform *wave);

/**
 * Resample a swift_waveform.
 *
 * @param wave    The swift_waveform to resample.
 * @param new_sps The sample rate to apply to the swift_waveform.
 * @return        SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if wave is
 *                NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_resample(swift_waveform *wave,
                                                      int new_sps);

/**
 * Convert a swift_waveform to a different encoding type.
 *
 * @param wave     The swift_waveform to convert
 * @param encoding One of:
 *                 - "pcm16" - 16-bit signed linear PCM
 *                 - "pcm8"  - 8-bit unsigned linear PCM
 *                 - "ulaw"  - µ-Law (8-bit by definition)
 *                 - "alaw"  - A-Law (8-bit by definition)
 * @return         SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if wave or
 *                 encoding is NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_convert(swift_waveform *wave,
                                                     const char *encoding);

/**
 * Get the raw sample data and byte-per-sample information about the data
 * from a swift_waveform.
 *
 * @param wave             The swift_waveform from which to get the sample
 *                         data and byte-per-sample information.
 * @param samples          Output - Buffer to receive the raw sample data.
 * @param nsamples         Output - Number of samples copied into the buffer.
 * @param bytes_per_sample Output - number of bytes-per-sample.
 * @return                 SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if wave
 *                         is NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_waveform_get_samples(const swift_waveform *wave, const void **samples,
                           int *nsamples, int *bytes_per_sample);

/**
 * Create a new swift_waveform and populate it with audio data from a buffer
 * of raw samples.
 *
 * @param samples   Buffer holding the raw data - must be pcm16 data.
 * @param nsamples  Number of samples.
 * @param frequency Frequency (sample rate) of data in the buffer.
 * @param channels  Number of channels.
 * @return          SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if wave is NULL.
 *
 * NOTE: If the samples and nsamples parameters are set to NULL and 0
 *       respectively, this will create an empty swift_waveform.
 **/
SWIFT_API
swift_waveform *  SWIFT_CALLCONV
swift_waveform_set_samples(const unsigned short *samples, const int nsamples,
                           const int frequency, const int channels);

/**
 * Set the number of channels for the given swift_waveform.
 *
 * @param wave     The swift_waveform for which to set channels.
 * @param channels Number of channels.  Must be 1 or 2.
 * @return         SWIFT_SUCCESS, or SWIFT_INVALID_PARAM.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_set_channels(swift_waveform *wave,
                                                          int channels);


/**
 * Append waveSrc swift_waveform to waveDst swift_waveform.
 *
 * @param waveDst The swift_waveform onto which to append waveSrc.
 * @param waveSrc The swift_waveform to append to waveDst.
 * @return        SWIFT_SUCCESS, or SWIFT_UNKNOWN_ERROR.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_waveform_concat(swift_waveform *waveDst,
                                                    swift_waveform *waveSrc);

/**
 * Return a string description of an error code.
 *
 * @param t The error code from which to retreive the string description.
 * @return  The string description of the error code.
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_strerror(swift_result_t t);

/**
 * Place a newly-allocated, comma-seperated list of all values for a given
 * parameter and given XML swift configuration file into the 'values'
 * output parameter.
 *
 * @param file    The path to the config file.
 * @param element The name of the parameter for which values should be
 *                returned.
 * @param data    Output - The newly-allocated, comma-seperated list of values.
 * @return        SWIFT_FILE_ERROR or SWIFT_SUCCESS.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_conffile_get_element_data(const char *file,
                                const char *element,
                                char **data);

/**
 * Retrieve the maximum number of concurrent synthesis calls allowed.
 *
 * @param engine        An active swift_engine structure.  Pass in the pointer
 *                      to the engine through which your swift_ports are open.
 * @param max_tokens    Output - the maximum number of concurrent synthesis
 *                      calls that can be made on this system.  A value of -1
 *                      indicates that the system is running with an Unlimited
 *                      Concurrency License.
 * @param tokens_in_use Output - the number of sysnthesis calls currently in
 *                      operation on this system at the time this call is made.
 *                      If max_tokens is 1 and tokens_in_use is -1, then the
 *                      system is running in Single Token Mode, in which the
 *                      license server is not used.
 * @return              SWIFT_SUCCESS on success, or SWIFT_NETWORK_ERROR if
 *                      communication to the license server cannot be made.  If
 *                      communication fails, max_tokens is set to 1 and 
 *                      tokens_in_use is set to -1, meaning "unknown."
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_license_get_concurrency_info(swift_engine *engine, int *max_tokens,
                                   int *tokens_in_use);

#ifdef __cplusplus
};
#endif /* __cplusplus */

#endif /* _SWIFT_H_ */

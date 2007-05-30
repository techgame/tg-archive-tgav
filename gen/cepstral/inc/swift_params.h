/* -*- c-file-style: "linux" -*- */
/* swift_params.h: Parameter and value objects for the Swift TTS engine.
 *
 * Copyright (c) 2003-2006 Cepstral LLC.  All rights reserved.
 *
 * Redistribution of this file, in whole or in part, with or without
 * modification, is not allowed without express written permission of
 * the copyright holder.
 */

#ifndef _SWIFT_PARAMS_H_
#define _SWIFT_PARAMS_H_

#include "swift_exports.h"
#include "swift_defs.h"

/**
 * \file swift_params.h
 * Parameter and value objects for the Swift TTS engine.
 **/

/**
 * Create a new parameter list.
 *
 * @param key The first parameter to set in the list.
 * @return    The newly created swift_params list.
 *
 * NOTE: There are optional parameters.  You can pass in as many keys as
 *       you'd like.
 **/
SWIFT_API
swift_params * SWIFT_CALLCONV swift_params_new(const char *key, ...);

/**
 * Claim ownership of the parameter list (increase its reference count).
 *
 * @param params The parameters for which to increase the reference count.
 * @return       The reference-counted swift_params that were passed in.
 *
 * Note that in all cases where a swift_params* is passed to a function in
 * swift.h, the Swift library claims ownership of the parameter list.
 * Therefore, you must call swift_params_ref() on it if you wish to use it
 * afterwards, and, conversely, you must not call swift_params_delete() on
 * it unless you have used swift_params_ref() to claim ownership.
 *
 * In practical terms, what this means is that you should just call
 * swift_params_new() in the argument list of these functions, like this:
 *
 * p = swift_open_port(swift_params_new(...));
 *
 * and not care about what happens to the object created, unless you really
 * need to know.
 **/
SWIFT_API
swift_params * SWIFT_CALLCONV swift_params_ref(swift_params *params);

/**
 * Relinquish ownership of a parameter list (decrease its reference count).
 *
 * @param params The parameters for which to decrease the reference count.
 * @return       The reference-decreased swift_params that were passed in.
 *
 * NOTE: This function decreases the reference count for the swift_params list.
 *       If the reference count reaches zero, the parameters are deleted.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_params_delete(swift_params *params);

/**
 * Dump contents of a parameter list to standard error (for debugging).
 *
 * @param params The swift_params list to dump to standard error.
 * @return       SWIFT_SUCCESS, or SWIFT_INVALID_POINTER if params is NULL.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_params_dump(swift_params *params);

/**
 * Set a value for a key.
 *
 * @param params The swift_params list in which to find the key to set.
 * @param name   The name of the key to set.
 * @param val    The value to set for the key.
 * @return       The string value of the value set for the key.
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_params_set_val(swift_params *params,
                                                 const char *name,
                                                 const swift_val *val);

/** Set an integer value in a swift_params **/
#define swift_params_set_int(p,n,i) \
    swift_params_set_val(p,n,swift_val_int((int)(i)))
/** Set a floating point value in a swift_params **/
#define swift_params_set_float(p,n,f) \
    swift_params_set_val(p,n,swift_val_float((float)(f)))
/** Set a string value in a swift_params **/
#define swift_params_set_string(p,n,s) \
    swift_params_set_val(p,n,swift_val_string((const char *)(s)))
/** Set a wide-character string value in a swift_params **/
#define swift_params_set_string16(p,n,s) \
    swift_params_set_val(p,n,swift_val_string16((const unsigned short *)(s)))

/**
 * User-Defined function that iterates over a parameter list.  Pointer to this
 * function is passed to swift_params_foreach().
 *
 * @param params The list of parameters over which to iterate.
 * @param name   The name of the current feature being evaluated.
 * @param val    The value of the current feature being evaluated.
 * @param udata  The user data pointer passed to swift_params_foreach().
 * @return       You should define your function to return SWIFT_SUCCESS
 *               until you no longer want to iterate.  Then return a non-
 *               SWIFT_SUCCESS value.
 **/
typedef swift_result_t (*swift_params_iterator)(swift_params *params,
                                                const char *name,
                                                swift_val *val,
                                                void *udata);
/**
 * Iterate over a parameter list.
 *
 * This function calls itor for each key/value pair in params,
 * stopping either when no more parameters remain, or if itor returns
 * a non-SWIFT_SUCCESS value.
 *
 * @param params The list of parameters over which to iterate.
 * @param itor   Pointer to the user-defined function for iterating over
 *               the params.
 * @param udata  User Data - pointer to anything you want to pass to the
 *               iterator function.
 * @return       SWIFT_SUCCESS if all key/value pair is itorated upon
 *               successfully, SWIFT_INVALID_POINTER if params is NULL, or
 *               the return value of itor if it is non-SWIFT_SUCCESS.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_params_foreach(swift_params *params,
                                                   swift_params_iterator itor,
                                                   void *udata);

/**
 * Search within a list of params for a key matching the given name.
 *
 * @param params The list of params in which to search for the key.
 * @param name   The name of the key for which to search.
 * @param def    Default value to return if the value for which you are
 *               searching is not found.
 * @return       The swift_val matching the name, if found.  If not found,
 *               the default value is returned.
 **/
SWIFT_API
const swift_val * SWIFT_CALLCONV
swift_params_get_val(const swift_params *params,
                     const char *name,
                     const swift_val *def);
/**
 * Search within a list of params for a key matching the given name.
 *
 * @param params The list of params in which to search for the key.
 * @param name   The name of the key for which to search.
 * @param def    The default string to return if the value for which you are
 *               searching is not found.
 * @return       The string value of the swift_val matching the given name,
 *               if found.  If not found, the default string is returned.
 *
 * NOTE: The value is returned as an 8-bit string (which is UTF-8 encoded if
 *       you created it with swift_val_string16, otherwise the encoding is
 *       unspecified).
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_params_get_string(const swift_params *params,
                                                    const char *name,
                                                    const char *def);

/**
 * Search within a list of params for a key matching the given name.
 *
 * @param params The list of params in which to search for the key.
 * @param name   The name of the key for which to search.
 * @param def    The default integer value to return if the value for which
 *               you are searching is not found.
 * @return       The integer value of the swift_val matching the name given,
 *               if found.  If not found, the default integer is returned.
 **/
SWIFT_API
int SWIFT_CALLCONV swift_params_get_int(const swift_params *params,
                                        const char *name,
                                        int def);

/**
 * Search within a list of params for a key matching the given name.
 *
 * @param params The list of params in which to search for the key.
 * @param name   The name of the key for which to search.
 * @param def    The default floating-point value to return if the value for
 *               which you are searching is not found.
 * @return       The floating-point value of the swift_val matching the name
 *               given, if found.  If not found, the default floating-point
 *               value is returned.
 **/
SWIFT_API
float SWIFT_CALLCONV swift_params_get_float(const swift_params *params,
                                            const char *name,
                                            float def);

/** Types of parameters that can be set in the engine, port, or voice. */
typedef enum {
    SWIFT_PARAM_NONE = -1, /**< Invalid parameter. */
    SWIFT_PARAM_FLAG,      /**< True or false (1/0). */
    SWIFT_PARAM_INT,       /**< Integer value. */
    SWIFT_PARAM_FLOAT,     /**< Floating point value. */
    SWIFT_PARAM_STRING,    /**< String value. */
    SWIFT_PARAM_ENUM       /**< Enumerated value. */
} swift_param_type_t;

/** Descriptor for an engine/port/voice parameter. */
typedef struct swift_param_desc {
    const char *name;        /**< Name of parameter (passed to set function). */
    const char *help;        /**< Textual description of parameter. */
    swift_param_type_t type; /**< Type of data accepted. */
    int undocumented;        /**< Undocumented. */
    const char **enum_vals;  /**< Enumeration values. */
    void *reserved[3];       /**< Reserved for future use. */
} swift_param_desc;

/** List of descriptors for valid parameters. */
SWIFT_API
extern const swift_param_desc swift_param_descriptors[];

/**
 * Validate a key/value pair based on swift_param_descriptors.
 *
 * @param name The name of the key to validate.
 * @param val  The value to validate against the descriptors.
 * @return     TRUE if the key/value pair is valid.  FALSE if not.
 **/
int swift_param_validate(const char *name,
                         const swift_val *val);

/**
 * Copy and validate parameters from one swift_params list to another.
 *
 * @param to   The list of swift_parms to which to copy.
 * @param from The list of swift_params from which to copy.
 * @return     SWIFT_SUCCESS if all parameters are copied successfully,<br>
 *             SWIFT_INVALID_PARAM if any of the parameters don't validate.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV
swift_params_set_params(swift_params *to,
                        const swift_params *from);

/**
 * Set one parameter in a parameter list, with validation.
 *
 * @param to   The list of params in which to set the parameter.
 * @param name The name of the parameter to set within the list of params.
 * @param val  The value of the parameter to set within the list of params.
 * @return     SWIFT_SUCCESS if the parameter is copied successfully,<br>
 *             SWIFT_INVALID_PARAM if the parameter does not validate.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_params_set_param(swift_params *to,
                                                     const char *name,
                                                     const swift_val *val);

/**
 * Create an integer-typed swift_val object.
 *
 * @param i The integer value to set within the swift_val.
 * @return  The newly-created integer-typed swift_val object.
 **/
SWIFT_API
swift_val * SWIFT_CALLCONV swift_val_int(int i);

/**
 * Create a floating-point-typed swift_val object.
 *
 * @param f The floating-point value to set within the swift_val.
 * @return  The newly-created floating-point-typed swift_val object.
 **/
SWIFT_API
swift_val * SWIFT_CALLCONV swift_val_float(float f);

/**
 * Create a character string-typed swift_val object.
 *
 * @param s The character string value to set within the swift_val.
 * @return  The newly-created character string-typed swift_val object.
 **/
SWIFT_API
swift_val * SWIFT_CALLCONV swift_val_string(const char *s);

/**
 * Create a UTF-16-encoded character string-typed swift_val object.
 *
 * @param s The UTF-16-encoded character string value to set within the
 *          swift_val.
 * @return  The newly-created UTF-16-encoded character string-typed swift_val
 *          object.
 **/
SWIFT_API
swift_val * SWIFT_CALLCONV swift_val_string16(const unsigned short *s);

/**
 * Claim ownership of the parameter list (increase its reference count).
 *
 * @param val The swift_val for which to increase the reference count.
 * @return    The reference-counted swift_val that was passed in.
 **/
SWIFT_API
swift_val * SWIFT_CALLCONV swift_val_ref(swift_val *val);

/**
 * Relinquish ownership of a swift_val object (decrease its reference count).
 *
 * @param val The swift_val for which to decrease the reference count.
 * @return    The reference-decreased swift_val that was passed in.
 *
 * NOTE: This function decreases the reference count for the swift_val.
 *       If the reference count reaches zero, the swift_val is deleted.
 **/
SWIFT_API
swift_result_t SWIFT_CALLCONV swift_val_delete(swift_val *val);

/**
 * Get the integer contents of a swift_val object.
 *
 * @param val The swift_val object from which to get the integer value
 * @return    The integer value of of the swift_val object
 **/
SWIFT_API
int SWIFT_CALLCONV swift_val_get_int(const swift_val *val);

/**
 * Get the floating-point contents of a swift_val object.
 *
 * @param val The swift_val object from which to get the floating-point value.
 * @return    The floating-point value of of the swift_val object
 **/
SWIFT_API
float SWIFT_CALLCONV swift_val_get_float(const swift_val *val);

/**
 * Get the string contents of a swift_val object.
 *
 * @param val The swift_val object from which to get the string value.
 *
 * Note that the swift_val object retains ownership of the string.
 *
 * If val is an integer or floating-point swift_val, it will be
 * silently "promoted" to a string.
 *
 * The contents are returned as an 8-bit string (which is UTF-8
 * encoded if you created it with swift_val_string16, otherwise the
 * encoding is unspecified).
 **/
SWIFT_API
const char * SWIFT_CALLCONV swift_val_get_string(const swift_val *val);

#endif /* _SWIFT_PARAMS_H_ */

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from _ctypes_cepstral import *
from swift_params import *
from swift_defs import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Code generated from:
#~   "inc/swift.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@bind(POINTER(swift_engine), [POINTER(swift_params)])
def swift_engine_open(params, _api_=None): 
    """swift_engine_open(params)
    
        params : POINTER(swift_params)
    """
    return _api_(params)
    

#~ line: 61, skipped: 10 ~~~~~~

@bind(swift_result_t, [POINTER(swift_engine)])
def swift_engine_close(engine, _api_=None): 
    """swift_engine_close(engine)
    
        engine : POINTER(swift_engine)
    """
    return _api_(engine)
    

#~ line: 86, skipped: 25 ~~~~~~

@bind(None, [POINTER(swift_engine), swift_voice_retention_policy_t])
def swift_engine_set_voice_retention_policy(engine, policy, _api_=None): 
    """swift_engine_set_voice_retention_policy(engine, policy)
    
        engine : POINTER(swift_engine)
        policy : swift_voice_retention_policy_t
    """
    return _api_(engine, policy)
    

#~ line: 103, skipped: 17 ~~~~~~

@bind(swift_voice_retention_policy_t, [POINTER(swift_engine)])
def swift_engine_get_voice_retention_policy(engine, _api_=None): 
    """swift_engine_get_voice_retention_policy(engine)
    
        engine : POINTER(swift_engine)
    """
    return _api_(engine)
    

#~ line: 117, skipped: 14 ~~~~~~

@bind(POINTER(swift_port), [POINTER(swift_engine), POINTER(swift_params)])
def swift_port_open(engine, params, _api_=None): 
    """swift_port_open(engine, params)
    
        engine : POINTER(swift_engine)
        params : POINTER(swift_params)
    """
    return _api_(engine, params)
    

#~ line: 127, skipped: 10 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port)])
def swift_port_close(port, _api_=None): 
    """swift_port_close(port)
    
        port : POINTER(swift_port)
    """
    return _api_(port)
    

#~ line: 142, skipped: 15 ~~~~~~

# typedef swift_callback_t
swift_callback_t = CFUNCTYPE(swift_result_t, POINTER(swift_event), swift_event_t, c_void_p)

#~ line: 160, skipped: 18 ~~~~~~

@bind(swift_callback_t, [POINTER(swift_port), swift_callback_t, c_uint, c_void_p])
def swift_port_set_callback(port, callback, mask, udata, _api_=None): 
    """swift_port_set_callback(port, callback, mask, udata)
    
        port : POINTER(swift_port)
        callback : swift_callback_t
        mask : c_uint
        udata : c_void_p
    """
    return _api_(port, callback, mask, udata)
    

#~ line: 169, skipped: 9 ~~~~~~

@bind(c_char_p, [swift_event_t])
def swift_event_type_get_name(type, _api_=None): 
    """swift_event_type_get_name(type)
    
        type : swift_event_t
    """
    return _api_(type)
    

#~ line: 178, skipped: 9 ~~~~~~

@bind(swift_event_t, [c_char_p])
def swift_event_name_get_type(name, _api_=None): 
    """swift_event_name_get_type(name)
    
        name : c_char_p
    """
    return _api_(name)
    

#~ line: 193, skipped: 15 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(POINTER(swift_port)), POINTER(swift_background_t)])
def swift_event_get_context(event, port, async, _api_=None): 
    """swift_event_get_context(event, port, async)
    
        event : POINTER(swift_event)
        port : POINTER(POINTER(swift_port))
        async : POINTER(swift_background_t)
    """
    return _api_(event, port, async)
    

#~ line: 207, skipped: 14 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(c_float), POINTER(c_float)])
def swift_event_get_times(event, start, len, _api_=None): 
    """swift_event_get_times(event, start, len)
    
        event : POINTER(swift_event)
        start : POINTER(c_float)
        len : POINTER(c_float)
    """
    return _api_(event, start, len)
    

#~ line: 224, skipped: 17 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(c_int), POINTER(c_int)])
def swift_event_get_textpos(event, start, len, _api_=None): 
    """swift_event_get_textpos(event, start, len)
    
        event : POINTER(swift_event)
        start : POINTER(c_int)
        len : POINTER(c_int)
    """
    return _api_(event, start, len)
    

#~ line: 236, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(c_char_p)])
def swift_event_get_text(event, text, _api_=None): 
    """swift_event_get_text(event, text)
    
        event : POINTER(swift_event)
        text : POINTER(c_char_p)
    """
    return _api_(event, text)
    

#~ line: 248, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(c_void_p), POINTER(c_int)])
def swift_event_get_audio(event, buf, nbytes, _api_=None): 
    """swift_event_get_audio(event, buf, nbytes)
    
        event : POINTER(swift_event)
        buf : POINTER(c_void_p)
        nbytes : POINTER(c_int)
    """
    return _api_(event, buf, nbytes)
    

#~ line: 264, skipped: 16 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(POINTER(swift_waveform)), c_int])
def swift_event_get_wave(event, wave, concatenate, _api_=None): 
    """swift_event_get_wave(event, wave, concatenate)
    
        event : POINTER(swift_event)
        wave : POINTER(POINTER(swift_waveform))
        concatenate : c_int
    """
    return _api_(event, wave, concatenate)
    

#~ line: 279, skipped: 15 ~~~~~~

@bind(swift_result_t, [POINTER(swift_event), POINTER(swift_result_t), POINTER(c_char_p)])
def swift_event_get_error(event, rv, errmsg, _api_=None): 
    """swift_event_get_error(event, rv, errmsg)
    
        event : POINTER(swift_event)
        rv : POINTER(swift_result_t)
        errmsg : POINTER(c_char_p)
    """
    return _api_(event, rv, errmsg)
    

#~ line: 302, skipped: 23 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), c_char_p, POINTER(swift_val), swift_background_t])
def swift_port_set_param(port, name, val, async, _api_=None): 
    """swift_port_set_param(port, name, val, async)
    
        port : POINTER(swift_port)
        name : c_char_p
        val : POINTER(swift_val)
        async : swift_background_t
    """
    return _api_(port, name, val, async)
    

#~ line: 315, skipped: 13 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), POINTER(swift_params), swift_background_t])
def swift_port_set_params(port, params, async, _api_=None): 
    """swift_port_set_params(port, params, async)
    
        port : POINTER(swift_port)
        params : POINTER(swift_params)
        async : swift_background_t
    """
    return _api_(port, params, async)
    

#~ line: 377, skipped: 62 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port), c_char_p, c_char_p])
def swift_port_find_first_voice(port, search_criteria, order_criteria, _api_=None): 
    """swift_port_find_first_voice(port, search_criteria, order_criteria)
    
        port : POINTER(swift_port)
        search_criteria : c_char_p
        order_criteria : c_char_p
    """
    return _api_(port, search_criteria, order_criteria)
    

#~ line: 388, skipped: 11 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port)])
def swift_port_find_next_voice(port, _api_=None): 
    """swift_port_find_next_voice(port)
    
        port : POINTER(swift_port)
    """
    return _api_(port)
    

#~ line: 398, skipped: 10 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port)])
def swift_port_rewind_voices(port, _api_=None): 
    """swift_port_rewind_voices(port)
    
        port : POINTER(swift_port)
    """
    return _api_(port)
    

#~ line: 409, skipped: 11 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), POINTER(swift_voice)])
def swift_port_set_voice(port, voice, _api_=None): 
    """swift_port_set_voice(port, voice)
    
        port : POINTER(swift_port)
        voice : POINTER(swift_voice)
    """
    return _api_(port, voice)
    

#~ line: 421, skipped: 12 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port), c_char_p])
def swift_port_set_voice_by_name(port, name, _api_=None): 
    """swift_port_set_voice_by_name(port, name)
    
        port : POINTER(swift_port)
        name : c_char_p
    """
    return _api_(port, name)
    

#~ line: 432, skipped: 11 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port), c_char_p])
def swift_port_set_voice_from_dir(port, dir, _api_=None): 
    """swift_port_set_voice_from_dir(port, dir)
    
        port : POINTER(swift_port)
        dir : c_char_p
    """
    return _api_(port, dir)
    

#~ line: 442, skipped: 10 ~~~~~~

@bind(POINTER(swift_voice), [POINTER(swift_port)])
def swift_port_get_current_voice(port, _api_=None): 
    """swift_port_get_current_voice(port)
    
        port : POINTER(swift_port)
    """
    return _api_(port)
    

#~ line: 470, skipped: 28 ~~~~~~

@bind(c_char_p, [POINTER(swift_voice), c_char_p])
def swift_voice_get_attribute(voice, attr, _api_=None): 
    """swift_voice_get_attribute(voice, attr)
    
        voice : POINTER(swift_voice)
        attr : c_char_p
    """
    return _api_(voice, attr)
    

#~ line: 485, skipped: 15 ~~~~~~

@bind(swift_result_t, [POINTER(swift_voice), POINTER(swift_params)])
def swift_voice_get_attributes(voice, out_params, _api_=None): 
    """swift_voice_get_attributes(voice, out_params)
    
        voice : POINTER(swift_voice)
        out_params : POINTER(swift_params)
    """
    return _api_(voice, out_params)
    

#~ line: 501, skipped: 16 ~~~~~~

@bind(swift_result_t, [POINTER(swift_voice), c_char_p])
def swift_voice_load_lexicon(voice, file, _api_=None): 
    """swift_voice_load_lexicon(voice, file)
    
        voice : POINTER(swift_voice)
        file : c_char_p
    """
    return _api_(voice, file)
    

#~ line: 511, skipped: 10 ~~~~~~

@bind(c_char_p, [POINTER(swift_port)])
def swift_port_language_encoding(port, _api_=None): 
    """swift_port_language_encoding(port)
    
        port : POINTER(swift_port)
    """
    return _api_(port)
    

#~ line: 524, skipped: 13 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), c_char_p])
def swift_port_load_sfx(port, file, _api_=None): 
    """swift_port_load_sfx(port, file)
    
        port : POINTER(swift_port)
        file : c_char_p
    """
    return _api_(port, file)
    

#~ line: 542, skipped: 18 ~~~~~~

@bind(swift_status_t, [POINTER(swift_port), swift_background_t])
def swift_port_status(port, async, _api_=None): 
    """swift_port_status(port, async)
    
        port : POINTER(swift_port)
        async : swift_background_t
    """
    return _api_(port, async)
    

#~ line: 557, skipped: 15 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), swift_background_t])
def swift_port_wait(port, async, _api_=None): 
    """swift_port_wait(port, async)
    
        port : POINTER(swift_port)
        async : swift_background_t
    """
    return _api_(port, async)
    

#~ line: 580, skipped: 23 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), swift_background_t, swift_event_t])
def swift_port_stop(port, async, place, _api_=None): 
    """swift_port_stop(port, async, place)
    
        port : POINTER(swift_port)
        async : swift_background_t
        place : swift_event_t
    """
    return _api_(port, async, place)
    

#~ line: 600, skipped: 20 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), swift_background_t, swift_event_t])
def swift_port_pause(port, async, place, _api_=None): 
    """swift_port_pause(port, async, place)
    
        port : POINTER(swift_port)
        async : swift_background_t
        place : swift_event_t
    """
    return _api_(port, async, place)
    

#~ line: 631, skipped: 31 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), c_void_p, c_int, c_char_p, POINTER(swift_background_t), POINTER(swift_params)])
def swift_port_speak_text(port, text, nbytes, encoding, async, params, _api_=None): 
    """swift_port_speak_text(port, text, nbytes, encoding, async, params)
    
        port : POINTER(swift_port)
        text : c_void_p
        nbytes : c_int
        encoding : c_char_p
        async : POINTER(swift_background_t)
        params : POINTER(swift_params)
    """
    return _api_(port, text, nbytes, encoding, async, params)
    

#~ line: 660, skipped: 29 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), c_char_p, c_char_p, POINTER(swift_background_t), POINTER(swift_params)])
def swift_port_speak_file(port, path, encoding, async, params, _api_=None): 
    """swift_port_speak_file(port, path, encoding, async, params)
    
        port : POINTER(swift_port)
        path : c_char_p
        encoding : c_char_p
        async : POINTER(swift_background_t)
        params : POINTER(swift_params)
    """
    return _api_(port, path, encoding, async, params)
    

#~ line: 694, skipped: 34 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), POINTER(c_int), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)])
def swift_port_get_perfstats(port, uttcount, avg_startup_time, audiotime, realtime, speed_ratio, _api_=None): 
    """swift_port_get_perfstats(port, uttcount, avg_startup_time, audiotime, realtime, speed_ratio)
    
        port : POINTER(swift_port)
        uttcount : POINTER(c_int)
        avg_startup_time : POINTER(c_float)
        audiotime : POINTER(c_float)
        realtime : POINTER(c_float)
        speed_ratio : POINTER(c_float)
    """
    return _api_(port, uttcount, avg_startup_time, audiotime, realtime, speed_ratio)
    

#~ line: 727, skipped: 33 ~~~~~~

@bind(c_char_p, [POINTER(swift_port), c_void_p, c_int, c_char_p, c_int, POINTER(swift_params)])
def swift_port_get_phones(port, text, nbytes, encoding, is_file, params, _api_=None): 
    """swift_port_get_phones(port, text, nbytes, encoding, is_file, params)
    
        port : POINTER(swift_port)
        text : c_void_p
        nbytes : c_int
        encoding : c_char_p
        is_file : c_int
        params : POINTER(swift_params)
    """
    return _api_(port, text, nbytes, encoding, is_file, params)
    

#~ line: 757, skipped: 30 ~~~~~~

@bind(POINTER(swift_waveform), [POINTER(swift_port), c_void_p, c_int, c_char_p, c_int, POINTER(swift_params)])
def swift_port_get_wave(port, text, nbytes, encoding, is_file, params, _api_=None): 
    """swift_port_get_wave(port, text, nbytes, encoding, is_file, params)
    
        port : POINTER(swift_port)
        text : c_void_p
        nbytes : c_int
        encoding : c_char_p
        is_file : c_int
        params : POINTER(swift_params)
    """
    return _api_(port, text, nbytes, encoding, is_file, params)
    

#~ line: 777, skipped: 20 ~~~~~~

@bind(swift_result_t, [POINTER(swift_port), POINTER(swift_waveform), POINTER(swift_background_t), POINTER(swift_params)])
def swift_port_play_wave(port, wav, async, params, _api_=None): 
    """swift_port_play_wave(port, wav, async, params)
    
        port : POINTER(swift_port)
        wav : POINTER(swift_waveform)
        async : POINTER(swift_background_t)
        params : POINTER(swift_params)
    """
    return _api_(port, wav, async, params)
    

#~ line: 785, skipped: 8 ~~~~~~

@bind(POINTER(swift_waveform), [])
def swift_waveform_new(_api_=None): 
    """swift_waveform_new()
    
        
    """
    return _api_()
    

#~ line: 817, skipped: 32 ~~~~~~

@bind(POINTER(swift_waveform), [c_char_p, c_char_p, c_char_p, c_int, c_int])
def swift_waveform_open(filename, format, encoding, sample_rate, num_channels, _api_=None): 
    """swift_waveform_open(filename, format, encoding, sample_rate, num_channels)
    
        filename : c_char_p
        format : c_char_p
        encoding : c_char_p
        sample_rate : c_int
        num_channels : c_int
    """
    return _api_(filename, format, encoding, sample_rate, num_channels)
    

#~ line: 826, skipped: 9 ~~~~~~

@bind(None, [POINTER(swift_waveform)])
def swift_waveform_print(wave, _api_=None): 
    """swift_waveform_print(wave)
    
        wave : POINTER(swift_waveform)
    """
    return _api_(wave)
    

#~ line: 835, skipped: 9 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform)])
def swift_waveform_close(wave, _api_=None): 
    """swift_waveform_close(wave)
    
        wave : POINTER(swift_waveform)
    """
    return _api_(wave)
    

#~ line: 857, skipped: 22 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), c_char_p, c_char_p])
def swift_waveform_save(wave, filename, format, _api_=None): 
    """swift_waveform_save(wave, filename, format)
    
        wave : POINTER(swift_waveform)
        filename : c_char_p
        format : c_char_p
    """
    return _api_(wave, filename, format)
    

#~ line: 866, skipped: 9 ~~~~~~

@bind(c_int, [POINTER(swift_waveform)])
def swift_waveform_get_sps(wave, _api_=None): 
    """swift_waveform_get_sps(wave)
    
        wave : POINTER(swift_waveform)
    """
    return _api_(wave)
    

#~ line: 880, skipped: 14 ~~~~~~

@bind(c_char_p, [POINTER(swift_waveform)])
def swift_waveform_get_encoding(wave, _api_=None): 
    """swift_waveform_get_encoding(wave)
    
        wave : POINTER(swift_waveform)
    """
    return _api_(wave)
    

#~ line: 892, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), c_int])
def swift_waveform_resample(wave, new_sps, _api_=None): 
    """swift_waveform_resample(wave, new_sps)
    
        wave : POINTER(swift_waveform)
        new_sps : c_int
    """
    return _api_(wave, new_sps)
    

#~ line: 908, skipped: 16 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), c_char_p])
def swift_waveform_convert(wave, encoding, _api_=None): 
    """swift_waveform_convert(wave, encoding)
    
        wave : POINTER(swift_waveform)
        encoding : c_char_p
    """
    return _api_(wave, encoding)
    

#~ line: 925, skipped: 17 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), POINTER(c_void_p), POINTER(c_int), POINTER(c_int)])
def swift_waveform_get_samples(wave, samples, nsamples, bytes_per_sample, _api_=None): 
    """swift_waveform_get_samples(wave, samples, nsamples, bytes_per_sample)
    
        wave : POINTER(swift_waveform)
        samples : POINTER(c_void_p)
        nsamples : POINTER(c_int)
        bytes_per_sample : POINTER(c_int)
    """
    return _api_(wave, samples, nsamples, bytes_per_sample)
    

#~ line: 943, skipped: 18 ~~~~~~

@bind(POINTER(swift_waveform), [POINTER(c_ushort), c_int, c_int, c_int])
def swift_waveform_set_samples(samples, nsamples, frequency, channels, _api_=None): 
    """swift_waveform_set_samples(samples, nsamples, frequency, channels)
    
        samples : POINTER(c_ushort)
        nsamples : c_int
        frequency : c_int
        channels : c_int
    """
    return _api_(samples, nsamples, frequency, channels)
    

#~ line: 954, skipped: 11 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), c_int])
def swift_waveform_set_channels(wave, channels, _api_=None): 
    """swift_waveform_set_channels(wave, channels)
    
        wave : POINTER(swift_waveform)
        channels : c_int
    """
    return _api_(wave, channels)
    

#~ line: 966, skipped: 12 ~~~~~~

@bind(swift_result_t, [POINTER(swift_waveform), POINTER(swift_waveform)])
def swift_waveform_concat(waveDst, waveSrc, _api_=None): 
    """swift_waveform_concat(waveDst, waveSrc)
    
        waveDst : POINTER(swift_waveform)
        waveSrc : POINTER(swift_waveform)
    """
    return _api_(waveDst, waveSrc)
    

#~ line: 975, skipped: 9 ~~~~~~

@bind(c_char_p, [swift_result_t])
def swift_strerror(t, _api_=None): 
    """swift_strerror(t)
    
        t : swift_result_t
    """
    return _api_(t)
    

#~ line: 992, skipped: 17 ~~~~~~

@bind(swift_result_t, [c_char_p, c_char_p, POINTER(c_char_p)])
def swift_conffile_get_element_data(file, element, data, _api_=None): 
    """swift_conffile_get_element_data(file, element, data)
    
        file : c_char_p
        element : c_char_p
        data : POINTER(c_char_p)
    """
    return _api_(file, element, data)
    

#~ line: 1016, skipped: 24 ~~~~~~

@bind(swift_result_t, [POINTER(swift_engine), POINTER(c_int), POINTER(c_int)])
def swift_license_get_concurrency_info(engine, max_tokens, tokens_in_use, _api_=None): 
    """swift_license_get_concurrency_info(engine, max_tokens, tokens_in_use)
    
        engine : POINTER(swift_engine)
        max_tokens : POINTER(c_int)
        tokens_in_use : POINTER(c_int)
    """
    return _api_(engine, max_tokens, tokens_in_use)
    


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ End of code generated from:
#~   "inc/swift.h"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


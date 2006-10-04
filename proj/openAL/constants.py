##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2006  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from raw import al, alc

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

alFormatMap = {
    'mono8': al.AL_FORMAT_MONO8, 
    'mono-8': al.AL_FORMAT_MONO8, 
    al.AL_FORMAT_MONO8: al.AL_FORMAT_MONO8,

    'mono16': al.AL_FORMAT_MONO16,
    'mono-16': al.AL_FORMAT_MONO16,
    al.AL_FORMAT_MONO16: al.AL_FORMAT_MONO16,

    'stereo8': al.AL_FORMAT_STEREO8,
    'stereo-8': al.AL_FORMAT_STEREO8,
    al.AL_FORMAT_STEREO8: al.AL_FORMAT_STEREO8,

    'stereo16': al.AL_FORMAT_STEREO16,
    'stereo-16': al.AL_FORMAT_STEREO16,
    al.AL_FORMAT_STEREO16: al.AL_FORMAT_STEREO16,
    }


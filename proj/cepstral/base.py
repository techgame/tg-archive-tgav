##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2007  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import weakref
from .raw import swift as _swift
from .raw.errors import CepstralError

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralObject(object):
    _as_parameter_ = None

    @classmethod
    def from_param(klass, as_parameter):
        self = klass.__new__(klass)
        self._set_param(as_parameter)
        return self

    def _set_param(self, as_parameter):
        if as_parameter:
            if self._as_parameter_:
                self.close()

            def closeObject(wr, as_parameter=as_parameter, closeFromParam=self._closeFromParam):
                closeFromParam(as_parameter)
            self._wr_close = weakref.ref(self, closeObject)

        self._as_parameter_ = as_parameter

    def close(self):
        self._closeFromParam(self)
        self._set_param(None)

    _closeFromParam = None


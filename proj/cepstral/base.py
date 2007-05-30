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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralObject(object):
    _as_parameter_ = None

    @classmethod
    def fromParam(klass, as_parameter):
        self = klass.__new__()
        self._setAsParam(as_parameter)
        return self

    def _setAsParam(self, as_parameter):
        if as_parameter:
            if self._as_parameter_:
                self.close()

            def closeObject(wr, as_parameter=as_parameter, closeFromParam=self._closeFromParam):
                closeFromParam(as_parameter)
            self._wr_close = weakref.ref(self, closeObject)

        self._as_parameter_ = as_parameter

    def close(self):
        self._closeFromParam(self)
        self._setAsParam(None)

    _closeFromParam = None


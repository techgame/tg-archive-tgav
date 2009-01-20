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

from .base import CepstralObject, _swift

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CepstralVoice(CepstralObject):
    def __repr__(self):
        return '<%s %s %s %s %s>' % (self.__class__.__name__,
                self.name, self.gender, self.age, 
                self.isLicensed() and "licensed" or "demo",
                )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def isLicensed(self):
        return bool(self.licenseKey)
    
    def getLicenseKey(self):
        return self.attrs.get('license/key')
    licenseKey = property(getLicenseKey)

    def getName(self):
        return self.attrs.get('speaker/name')
    name = property(getName)

    def getAge(self):
        return self.attrs.get('speaker/age')
    age = property(getAge)

    def getGender(self):
        return self.attrs.get('speaker/gender')
    gender = property(getGender)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    _attrs = None
    def allAttrs(self, reload=False):
        attrs = self._attrs
        if attrs is None or reload:
            attrs = self.loadAttrs()
            self._attrs = attrs
        return attrs
    attrs = property(allAttrs)

    def loadAttrs(self):
        attrs = {}

        params = _swift.swift_params_new(None)
        _swift.swift_voice_get_attributes(self, params)

        @_swift.swift_params_iterator
        def iterParam(params, name, val, udata):
            attrs[str(name)] = str(_swift.swift_val_get_string(val))
            return 0

        _swift.swift_params_foreach(params, iterParam, None)
        _swift.swift_params_delete(params)

        return attrs

    def loadLexicon(self, filename):
        _swift.swift_voice_load_lexicon(self, filename)

    @staticmethod
    def _closeFromParam(param):
        pass


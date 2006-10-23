##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~ Copyright (C) 2002-2005  TechGame Networks, LLC.              ##
##~                                                               ##
##~ This library is free software; you can redistribute it        ##
##~ and/or modify it under the terms of the BSD style License as  ##
##~ found in the LICENSE file included with this distribution.    ##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from OpenGL import GL
from attributeMgr import AttributeChangeElement, ClientAttributeChangeElement

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StateManagerBase(object):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ Definitions 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __init__(self, statevector=None):
        if statevector is not None:
            self._statevector = statevector
        else: self._statevector = {}

    def _updateState(self, state, enabled):
        self._statevector[state] = enabled

    def getState(self, state):
        return self._statevector.get(state, None)
    def setState(self, state, enabled):
        current = self._statevector.get(state, None)
        if current != enabled:
            self._setState(state, enabled)
            self._updateState(state, enabled)
    def delState(self, state):
        try: del self._statevector[state]
        except KeyError: pass

    def reset(self, state=None):
        if state is None:
            self._statevector.clear()
        else: self.delState(state)

    def isEnabled(self, state): 
        return self.getState(state) == 1
    def isNotEnabled(self, state): 
        return self.getState(state) != 1
    def enable(self, state): 
        self.setState(state, 1)

    def isDisabled(self, state): 
        return self.getState(state) == 0
    def isNotDisabled(self, state): 
        return self.getState(state) != 0
    def disable(self, state): 
        self.setState(state, 0)

    def isKnown(self, state): 
        return self.getState(state) != None
    def isUnknown(self, state): 
        return self.getState(state) == None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StateManagerStats(object):
    _statechanges = None

    def _updateState(self, state, enabled):
        sc = self._statechanges
        if sc is None: self._statechanges = {state: 1}
        else: sc[state] = sc.get(state, 0) + 1

        self._statevector[state] = enabled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StateManager(StateManagerBase):
    attributeChange = AttributeChangeElement(GL.GL_ENABLE_BIT)

    def _setState(self, state, enabled):
        if enabled: 
            GL.glEnable(state)
        else: 
            GL.glDisable(state)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ClientStateManager(StateManagerBase):
    clientAttributeChange = ClientAttributeChangeElement(GL.GL_CLIENT_VERTEX_ARRAY_BIT)

    def _setState(self, state, enabled):
        if enabled: 
            GL.glEnableClientState(state)
        else: 
            GL.glDisableClientState(state)


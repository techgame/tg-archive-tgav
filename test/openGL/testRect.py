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

import unittest

from numpy import allclose

from TG.openGL.data import Rect

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TestRect(unittest.TestCase):
    def doVTest(self, rawQuestion, rawAnswer):
        #question = [rawQuestion.pos.tolist(), rawQuestion.size.tolist()]
        #answer = rawAnswer
        #self.assertEqual(question, answer)
        self.failUnless(allclose(rawQuestion.pos, rawAnswer[0]), (rawQuestion.pos.tolist(), rawAnswer[0]))
        self.failUnless(allclose(rawQuestion.size, rawAnswer[1]), (rawQuestion.size.tolist(), rawAnswer[1]))

    def testBasic(self):
        self.doVTest(Rect(), [[0., 0., 0.], [1., 1., 0.]])

        r = Rect()
        r.size = [3.125, 6.25]
        self.doVTest(r, [[0., 0., 0.], [3.125, 6.25, 0.]])

        r = Rect(dtype='B')
        r.size = [3.125, 6.25]
        self.doVTest(r, [[0, 0, 0], [3, 6, 0]])

    def testFromSize(self):
        self.doVTest(Rect.fromSize((3.125, 6.25), dtype='f'), [[0, 0, 0], [3.125, 6.25, 0]])
        self.doVTest(Rect.fromSize((3.125, 6.25), dtype='B'), [[0, 0, 0], [3, 6, 0]])
    
    def testFromSizeAspect(self):
        self.doVTest(Rect.fromSize((3.125, 6.25), 1.5, dtype='f'), [[0, 0, 0], [3.125, 3.125/1.5, 0]])
        self.doVTest(Rect.fromSize((3.125, 6.25), 1.5, dtype='B'), [[0, 0, 0], [3, int(3/1.5), 0]])
    
    def testFromPosSize(self):
        self.doVTest(Rect.fromPosSize((2.3, 4.8), (3.125, 6.25), dtype='f'), [[2.3, 4.8, 0], [3.125, 6.25, 0]])
        self.doVTest(Rect.fromPosSize((2.3, 4.8), (3.125, 6.25), dtype='B'), [[2, 4, 0], [3, 6, 0]])
    
    def testFromPosSizeAspect(self):
        self.doVTest(Rect.fromPosSize((2.3, 4.8), (3.125, 6.25), 1.5, dtype='f'), [[2.3, 4.8, 0], [3.125, 3.125/1.5, 0]])
        self.doVTest(Rect.fromPosSize((2.3, 4.8), (3.125, 6.25), 1.5, dtype='B'), [[2, 4, 0], [3, 2, 0]])
    
    def testFromCorners(self):
        self.doVTest(Rect.fromCorners((2.5, 1.5), (5.75, 6.75), dtype='f'), [[2.5, 1.5, 0], [3.25, 5.25, 0]])
        self.doVTest(Rect.fromCorners((2.5, 1.5), (5.75, 6.75), dtype='B'), [[2, 1, 0], [3, 5, 0]])
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Unittest Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    unittest.main()


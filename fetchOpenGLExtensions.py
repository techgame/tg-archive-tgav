#!/usr/bin/env python
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Imports 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import sys
import os
import urllib
from  htmllib import HTMLParser
from formatter import NullFormatter

import robotparser 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Definitions 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class OpenGLRegistrySpecsLinkGrabber(HTMLParser):
    url = 'http://opengl.org/registry/'
    registrySpecLink = "/registry/specs/"

    def __init__(self, url=url):
        self.url = url
        HTMLParser.__init__(self, NullFormatter())

        if self.url:
            self.parseURL(self.url)

    def parseFile(self, htmlFile):
        buf = True
        while buf:
            buf = htmlFile.read(1024)
            self.feed(buf)
        self.close()

    def parseURL(self, url):
        robotsURL = urllib.basejoin(url, '/robots.txt')
        self.robots = robotparser.RobotFileParser(robotsURL)
        if self._robotCanFetch(url):
            self.parseFile(urllib.urlopen(url))

    def _robotCanFetch(self, url):
        return self.robots.can_fetch(type(self).__name__, url)

    def anchor_bgn(self, href, name, type):
        href = urllib.basejoin(self.url, href)
        parts = href.split(self.registrySpecLink, 1)
        extPath = ''.join(parts[1:2])
        if extPath:
            self.fetchExtension(href, extPath)

    def fetchExtension(self, extHref, extPath):
        assert not '..' in extPath
        assert not extPath.startswith('/')

        if extPath.startswith('doc'):
            return False

        extPath = os.path.join('specs', extPath)
        if os.path.exists(extPath):
            return False

        if not self._robotCanFetch(url):
            print 'Robot not allowed to download:', extPath
            return False

        extBase = os.path.split(extPath)[0]
        if not os.path.exists(extBase):
            os.makedirs(extBase)

        print 'Downloading:', extPath
        urllib.urlretrieve(extHref, extPath)
        return True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Main 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__=='__main__':
    lg = OpenGLRegistrySpecsLinkGrabber()


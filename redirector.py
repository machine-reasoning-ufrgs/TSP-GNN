import os
import sys
import tempfile

STDOUT = 1
STDERR = 2

class Redirector(object):
    def __init__(self, fd=STDOUT):
        self.fd = fd
        self.started = False

    def start(self):
        if not self.started:
            self.tmpfd, self.tmpfn = tempfile.mkstemp()

            self.oldhandle = os.dup(self.fd)
            os.dup2(self.tmpfd, self.fd)
            os.close(self.tmpfd)

            self.started = True

    def flush(self):
        if self.fd == STDOUT:
            sys.stdout.flush()
        elif self.fd == STDERR:
            sys.stderr.flush()

    def stop(self):
        if self.started:
            self.flush()
            os.dup2(self.oldhandle, self.fd)
            os.close(self.oldhandle)
            tmpr = open(self.tmpfn, 'rb')
            output = tmpr.read()
            tmpr.close()  # this also closes self.tmpfd
            os.unlink(self.tmpfn)

            self.started = False
            return output
        else:
            return None

class RedirectorOneFile(object):
    def __init__(self, fd=STDOUT):
        self.fd = fd
        self.started = False
        self.inited = False

        self.initialize()

    def initialize(self):
        if not self.inited:
            self.tmpfd, self.tmpfn = tempfile.mkstemp()
            self.pos = 0
            self.tmpr = open(self.tmpfn, 'rb')
            self.inited = True

    def start(self):
        if not self.started:
            self.oldhandle = os.dup(self.fd)
            os.dup2(self.tmpfd, self.fd)
            self.started = True

    def flush(self):
        if self.fd == STDOUT:
            sys.stdout.flush()
        elif self.fd == STDERR:
            sys.stderr.flush()

    def stop(self):
        if self.started:
            self.flush()
            os.dup2(self.oldhandle, self.fd)
            os.close(self.oldhandle)
            output = self.tmpr.read()
            self.pos = self.tmpr.tell()
            self.started = False
            return output
        else:
            return None

    def close(self):
        if self.inited:
            self.flush()
            self.tmpr.close()  # this also closes self.tmpfd
            os.unlink(self.tmpfn)
            self.inited = False
            return output
        else:
            return None



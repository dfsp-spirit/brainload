"""
Custom errors for brainload.
"""

class HemiFileIOError(IOError):
    """
    A special IOError that holds information on the hemisphere the file that caused the error belonges to.

    This allows users to see which hemisphere caused issues when they loaded both at once.
    """
    def __init__(self, errno, message, filename, hemi, *args):
        self.message = message
        self.hemi = hemi
        super(HemiFileIOError, self).__init__(errno, message, filename, *args)

    def __str__(self):
        return '[Errno ' + str(self.errno) + '] ' + self.message + ': \'' + self.filename + '\'' + ' for hemi \'' + self.hemi + '\''

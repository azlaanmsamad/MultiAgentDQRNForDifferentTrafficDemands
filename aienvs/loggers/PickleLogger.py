from aienvs.listener.Listener import Listener
import pickle
from _io import BytesIO


class PickleLogger(Listener):
    
    """
    Logs final results coming from a DefaultRunner to a pickle binary file
    """
    
    def __init__(self, outstream: BytesIO):
        """
        @param outstream a general outputstream, either file or StringIO.
        Create with  open("myfile.txt", "r", encoding="utf-8") or
        io.StringIO("some initial text data")
        """
        self._outstream = outstream
        
    def notifyChange(self, data):
        pickle.dump(data, self._outstream)
        

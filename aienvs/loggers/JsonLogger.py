from aienvs.listener.Listener import Listener
import json
from _pyio import TextIOBase
import numpy as np


def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.int64): 
        return int(obj)  

    return obj.__dict__


class JsonLogger(Listener):
    
    """
    Logs final results coming from a DefaultRunner to a json file.
    """
    
    def __init__(self, outstream: TextIOBase):
        """
        @param outstream a general outputstream, either file or StringIO.
        Create with  open("myfile.txt", "r", encoding="utf-8") or
        io.StringIO("some initial text data")
        """
        self._outstream = outstream

    # Override
    def notifyChange(self, data):
        self._outstream.writelines(json.dumps(data, default=serialize))
        self._outstream.writelines('\n')
 

from aienvs.listener.Listenable import Listenable
from aienvs.listener.Listener import Listener
import traceback


class DefaultListenable(Listenable):
    """
    Default implementation for Listenable
    """

    def __init__(self):
        self._listeners = []
    
    # Override
    def addListener(self, l:Listener):
        self._listeners.append(l)
        
    # Override
    def removeListener(self, l: Listener):
        self._listeners.remove(l)

    def notifyAll(self, data):
        """
        Notifies all listeners of a change.
        This should only be called by the owner of this, not by 
        listeners or others.
        Listeners should not throw. But as courtesy, any exceptions 
        will be caught and printed out. 
        @param data the information about the change
        """
        for l in self._listeners:
            try:
                l.notifyChange(data)
            except:
                print(traceback.format_exc())

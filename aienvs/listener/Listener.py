from abc import ABC, abstractmethod


class Listener(ABC):

    @abstractmethod
    def notifyChange(self, data):
        """
        A notification call that notifies the Listener that something changed in
        the object being listened to. Consult the listened object for details on the 
        received data.
        
        NOTICE Notifications run in the thread of the caller which is the object being
        listened to. The caller will be blocked until this call to notifyChange returns.
        Therefore callbacks must return quickly and do only light processing.
        
        @param data additional data, typically the new value associated with the event. 
        """
        pass

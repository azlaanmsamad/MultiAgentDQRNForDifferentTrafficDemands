from aienvs.listener.Listener import Listener


# we don't use ABC because python then errs "can't resolve MRO"
class Listenable():

    def addListener(self, l: Listener):
        """
        @param l a Listener to be added 
        """
        pass

    def removeListener(self, l: Listener):
        """
        @param l a Listener to be removed 
        """
        pass
     

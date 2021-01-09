from collections import OrderedDict
from abc import ABC, abstractmethod
from gym.spaces import Dict


class ModifiedActionSpace(ABC, Dict):
    '''
    abstract class for action space modifiers.
    It takes an existing actionspace (space.Dict)
    and presents it as a modified one
    '''
    
    @abstractmethod
    def unpack(self, action: OrderedDict) -> OrderedDict:
        '''
        @return: the action converted back to its original space
        '''
        pass
    
    @abstractmethod
    def getOriginalSpace(self) -> OrderedDict: 
        '''
        The original actionspace 
        '''
        pass
